import torch

from copy import deepcopy
from time import time
from tqdm import tqdm

from models.majority_vote import MultipleMajorityVote

def train_batch(n, data, model, optimizer, bound=None, loss=None, nb_iter=1e4, monitor=None):

    model.train()

    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:
        optimizer.zero_grad()

        if bound is not None:
            cost = bound(n, model, model.risk(data, loss))

        else:
            cost = model.risk(data, loss)

        pbar.set_description("train obj %s" % cost.item())

        cost.backward()
        optimizer.step()
        
        #if monitor:
        #    monitor.write_all(i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})

def train_stochastic(dataloader, model, optimizer, epoch, bound=None, loss=None, monitor=None, model_type=None, order=0.):

    model.train()

    last_iter = epoch * len(dataloader)
    train_obj = 0.

    pbar = tqdm(dataloader)

    for i, batch in enumerate(pbar):

        n = len(batch[0])
        data = batch[1], model(batch[0])

        optimizer.zero_grad()

        if bound is not None:
            cost = bound(n, model, model.risk(data, loss))
        else:
            cost = model.risk(data, loss)
        train_obj += cost.item()
        pbar.set_description("avg train obj %f" % (train_obj / (i + 1)))

        cost.backward()
        optimizer.step()
        #add clamping of the posterior when renyi dirichlet
        if model_type == "des_dir": 
            with torch.no_grad() :
                thresh = (order-1)/order * model.prior
                model.post.clamp_(min=torch.log(thresh + 1e-4))
            
        #if monitor:
        #    monitor.write_all(last_iter+i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})

def train_stochastic_multiset(dataloaders, model, optimizer, epoch, bound=None, loss=None, monitor=None,  model_type=None, order=0.):

    model.train()

    last_iter = epoch * len(dataloaders[0])
    train_obj = 0.

    pbar = tqdm(range(len(dataloaders[0])))
    
    for i, *batches in zip(pbar, *dataloaders):
        X = [batch[0] for batch in batches]
        # sum sizes of loaders
        n = sum(map(len, X))
        pred = model(X)
        data = [(batches[i][1], pred[i]) for i in range(len(batches))]

        optimizer.zero_grad()

        if bound is not None:
            cost = bound(n, model, model.risk(data, loss))

        else:
            cost = model.risk(data, loss)

        train_obj += cost.item()

        pbar.set_description("avg train obj %f" % (train_obj / (i + 1)))

        cost.backward()
        optimizer.step()
        i=0
        if model_type == "des_dir":
            with torch.no_grad():
                for mv in model.mvs:
                    i+=1
                    thresh = (order - 1) / order * mv.prior
                    mv.post.clamp_(min=torch.log(thresh + 1e-4))
                    
        #if monitor:
        #    monitor.write_all(last_iter+i, model.get_post(), model.get_post_grad(), train={"Train-obj": cost.item()})


def evaluate(dataloader, model, epoch=-1, bounds=None, loss=None, monitor=None, tag="val", best_post=None):
#def evaluate(dataloader, model, epoch=-1, bounds=None, loss=None, monitor=None, tag="val"):
    model.eval()
    risk = 0.
    strength = 0.
    n = 0

    for batch in dataloader:

        data = batch[1], model(batch[0])
        risk += model.risk(data, loss=loss, mean=False, post=best_post)
        strength += sum(model.voter_strength(data))
        n += len(data[0])

    risk /= n
    strength /= n
    total_metrics = {"error": risk.item(), "strength": strength.item()}
    if bounds is not None:

        for k in bounds.keys():
            total_metrics[k] = bounds[k](n, model, risk).item()

    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics

def evaluate_multiset(dataloaders, model, epoch=-1, bounds=None, loss=None, monitor=None, tag="val", best_post=None):
   
    model.eval()

    risk = 0.
    n = 0
    strength = 0.

    for batches in zip(*dataloaders):

        X = [batch[0] for batch in batches]
        pred = model(X)
        data = [(batches[i][1], pred[i]) for i in range(len(batches))]
        risk += model.risk(data, loss=loss, mean=False, posts=best_post)
        strength += sum(model.voter_strength(data))

        n += len(X[0])

    risk /= n
    strength /= n
    total_metrics = {"error": risk.item(), "strength": strength.item()}
    if bounds is not None:
        if list(bounds.values())[0] is not None:
            for k in bounds.keys():
                total_metrics[k] = bounds[k](n, model, risk).item()
        else :
            for k in bounds.keys():
                total_metrics[k] = risk 
        
    if monitor:
        monitor.write(epoch, **{tag: total_metrics})

    return total_metrics


def select_best_individual_vote(
    dataloaders,
    testloader,
    model,
    bound,
    bound_type,
    loss=None,
    best_posts=None,
):
    """Select the individual majority vote with the smallest training bound.

    Each vote is evaluated only on its associated cross-fitting split.  ``bound``
    must therefore use the sample size passed to it, rather than a sample size
    captured from the full training set.
    """
    if not isinstance(model, MultipleMajorityVote):
        raise TypeError("individual selection requires a MultipleMajorityVote")
    if len(dataloaders) != len(model.mvs):
        raise ValueError("one dataloader is required for each majority vote")
    if best_posts is not None and len(best_posts) != len(model.mvs):
        raise ValueError("one posterior draw is required for each majority vote")

    individual_stats = []
    for index, (dataloader, majority_vote) in enumerate(zip(dataloaders, model.mvs)):
        post = None if best_posts is None else best_posts[index]
        stats = evaluate(
            dataloader,
            majority_vote,
            bounds={bound_type: bound},
            loss=loss,
            best_post=post,
        )
        individual_stats.append(stats)

    selected_index = min(
        range(len(individual_stats)),
        key=lambda index: individual_stats[index][bound_type],
    )
    selected_model = deepcopy(model.mvs[selected_index])
    selected_post = None if best_posts is None else best_posts[selected_index].detach().clone()
    selected_test_stats = evaluate(
        testloader,
        selected_model,
        loss=loss,
        best_post=selected_post,
    )

    return selected_model, selected_index, individual_stats, selected_test_stats


def stochastic_routine(trainloader, testloader, model, optimizer, bound, bound_type, loss=None, monitor=None, num_epochs=100, lr_scheduler=None, model_type=None, order=None, pretraining = False):

    best_bound = float("inf")
    best_e = -1
    no_improv = 0
    best_train_stats = None
    best_model = None
    best_thetas = None

    if isinstance(model, MultipleMajorityVote): # then expect multiple dataloaders
        train_routine = train_stochastic_multiset
        val_routine = evaluate_multiset
        test_routine = lambda d, *args, **kwargs: evaluate_multiset((d, d), *args, **kwargs)
    else:
        train_routine, val_routine, test_routine = train_stochastic, evaluate, evaluate

    
    t1 = time()
    for e in range(num_epochs):
        
        train_routine(trainloader, model, optimizer, epoch=e, bound=bound, loss=loss, monitor=monitor, model_type=model_type, order=order)

        current_thetas = model.get_theta(draw=True)
        if "des" in model_type : 
            train_stats = val_routine(trainloader, model, epoch=e, bounds={bound_type: bound}, loss=loss, monitor=monitor, tag="train", best_post=current_thetas)
        else :
            train_stats = val_routine(trainloader, model, epoch=e, bounds={bound_type: bound}, loss=loss, monitor=monitor, tag="train")
        no_improv += 1
        
        if train_stats[bound_type] < best_bound:
            no_improv = 0
            best_bound = train_stats[bound_type]
            best_train_stats = train_stats
            best_e = e
            best_model = deepcopy(model)
            best_thetas = current_thetas.detach().clone()
            
        # reduce learning rate if needed
        if lr_scheduler:
            lr_scheduler.step(train_stats[bound_type])

        if no_improv >= max(1, num_epochs // 4):
            break

    t2 = time()

    train_error = val_routine(trainloader, best_model, best_post = best_thetas) 
                                                        
    test_error = test_routine(testloader, best_model, best_post = best_thetas)
    
    if pretraining :
        print(f"Test error: {test_error['error']}; train error: {train_error}\n")
    else :
        print(f"Test error: {test_error['error']}; {bound_type} bound: {best_train_stats[bound_type]}\n")

    # Keep the historical return signature while exposing the draw associated
    # with the selected epoch for post-training individual-vote selection.
    best_model.best_thetas = best_thetas
    return best_model, best_bound, best_train_stats, train_error, test_error, t2-t1
