import hydra
from pathlib import Path

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cbound import cbound_routine
from core.bounds import BOUNDS
from core.losses import sigmoid_loss, moment_loss, rand_loss
from core.monitors import MonitorMV
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MultipleMajorityVote, MajorityVote
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps

from optimization import select_best_individual_vote, stochastic_routine

@hydra.main(config_path='config/training.yaml')
def main(cfg):

    if cfg.training.risk == "CB":
        # the C-bound comes with its own certificate: PAC-Bound 2 of Germain et al. (2015)
        cfg.bound.type = "cbound"

    ROOT_DIR = f"{hydra.utils.get_original_cwd()}/results/{cfg.dataset}/{cfg.training.risk}/{cfg.bound.type}/optimize-bound={cfg.training.opt_bound}/stochastic-bound={cfg.bound.stochastic}/{cfg.model.pred}/learnt_prior={cfg.pretraining.learnt_prior}/M={cfg.model.M}/max-depth={cfg.model.tree_depth}/prior={cfg.model.prior}/"
    ROOT_DIR = Path(ROOT_DIR)
    div ="KL"
    if cfg.model.uniform:
        ROOT_DIR /= "uniform"
    else:
        ROOT_DIR /= f"lr={cfg.training.lr}/batch-size={cfg.training.batch_size}/"

    if cfg.training.risk == "MC":
        ROOT_DIR /= f"MC={cfg.training.MC_draws}"
    elif (cfg.training.risk == "des_dir"):
        ROOT_DIR /= f"order={cfg.training.order}"
        ROOT_DIR /= f"MC={cfg.training.MC_draws}"
        ROOT_DIR /=f"risk-type={cfg.training.risk_type}"
        div = "Renyi"
    elif (cfg.training.risk == "des_rivasplata_dir") :
        ROOT_DIR /= f"MC={cfg.training.MC_draws}"
        ROOT_DIR /=f"risk-type={cfg.training.risk_type}"

    
    # define params for each method
    risks = { # type: (loss, bound-coeff, distribution-type, kl factor)
        "exact": (None, 1., "dirichlet", 1.),
        "MC": (lambda x, y, z: sigmoid_loss(x, y, z, c=cfg.training.sigmoid_c), 1., "dirichlet", 1.),
        "Rnd": (lambda x, y, z: rand_loss(x, y, z, n=cfg.training.rand_n), 2., "categorical", cfg.training.rand_n),
        "FO": (lambda x, y, z: moment_loss(x, y, z, order=1), 2., "categorical", 1.),
        "SO": (lambda x, y, z: moment_loss(x, y, z, order=2), 4., "categorical", 2.),
        "des_dir" : (lambda x, y, z: sigmoid_loss(x, y, z, c=cfg.training.sigmoid_c), 1., "dirichlet", 1.),
        "des_rivasplata_dir" : (lambda x, y, z: sigmoid_loss(x, y, z, c=cfg.training.sigmoid_c), 1., "dirichlet", 1.),
    }

    train_errors, test_errors, train_losses, bounds, strengths, entropies, kls, times = [], [], [], [], [], [], [], []
    selected_votes, individual_bounds, cross_bounds, cross_test_errors = [], [], [], []
    for i in range(cfg.num_trials):
        
        deterministic(cfg.training.seed+i)

        SAVE_DIR = ROOT_DIR / f"seed={cfg.training.seed+i}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        result_file = SAVE_DIR / "err-b.npy"
        cached_results = None
        if result_file.is_file():
            cached_results = np.load(result_file, allow_pickle=True).item()

        selection_is_cached = cfg.model.pred != "rf" or "selected-vote" in (cached_results or {})
        if cached_results is not None and selection_is_cached:
            # load saved stats
            seed_results = cached_results

        elif cfg.training.risk == "CB":

            data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data", valid_size=0)
            seed_results = cbound_routine(cfg, data)
            np.save(result_file, seed_results)

        else:

            loss, coeff, distr, kl_factor = risks[cfg.training.risk]
            seed_results = {}

            data = Dataset(cfg.dataset, normalize=True, data_path=Path(hydra.utils.get_original_cwd()) / "data", valid_size=0)

            m = 0
            if cfg.model.pred == "stumps-uniform":
                predictors, M = uniform_decision_stumps(cfg.model.M, data.X_train.shape[1], data.X_train.min(0), data.X_train.max(0), distr )

            elif cfg.model.pred == "rf": # random forest

                if cfg.model.tree_depth == "None":
                    cfg.model.tree_depth = None

                predictors, M = two_forests(cfg.model.M, 0.5, data.X_train, data.y_train, max_samples=cfg.model.bootstrap, max_depth=cfg.model.tree_depth, binary=data.binary)

            else:
                raise NotImplementedError("model.pred should be one the following: [stumps-uniform, rf]")


            bound = None
            if cfg.training.opt_bound:

                if cfg.bound.stochastic:
                    bound = lambda n, model, risk: BOUNDS[cfg.bound.type](n, model, risk, delta=cfg.bound.delta, coeff=coeff, order=cfg.training.order, div = div, type=cfg.training.risk)

                else:
                    n = len(data.X_train)
                    bound = lambda _, model, risk: BOUNDS[cfg.bound.type](n, model, risk, delta=cfg.bound.delta, coeff=coeff, order=cfg.training.order, div = div, type=cfg.training.risk)

            if cfg.model.pred == "rf": # a loader per posterior

                m_train = len(data.X_train) // 2
                train1 = TorchDataset(data.X_train[m_train:], data.y_train[m_train:])
                train2 = TorchDataset(data.X_train[:m_train], data.y_train[:m_train])
                trainloader = [
                    DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                    DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                ] 

            else:
                train = TorchDataset(data.X_train, data.y_train)
                trainloader = DataLoader(train, batch_size=cfg.training.batch_size, num_workers=cfg.num_workers, shuffle=True)

            testloader = DataLoader(TorchDataset(data.X_test, data.y_test), batch_size=4096, num_workers=cfg.num_workers, shuffle=False)

            if cfg.model.pred == "rf":
                betas = [torch.ones(M) * cfg.model.prior for p in predictors] # prior

                model = MultipleMajorityVote(predictors, betas, weights=(0.5, 0.5), mc_draws=cfg.training.MC_draws, distr=distr, kl_factor=kl_factor, risk_type=cfg.training.risk_type, order=cfg.training.order)
            
            else:
                betas = torch.ones(M) * cfg.model.prior # prior

                model = MajorityVote(predictors, betas, mc_draws=cfg.training.MC_draws, distr=distr, kl_factor=kl_factor, risk_type=cfg.training.risk_type, order=cfg.training.order)

            
            
            if cfg.model.pred == "rf" and cfg.pretraining.learnt_prior == True :
                bound_prior = None 
                betas = [torch.ones(M) * cfg.model.prior for p in predictors] # prior

                pretrainloader = [
                    DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                    DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                ] 

                
                pre_train_model = MultipleMajorityVote(
                    predictors,
                    betas,
                    weights=(0.5, 0.5),
                    mc_draws=cfg.training.MC_draws,
                    distr=distr,
                    kl_factor=kl_factor,
                    risk_type=cfg.training.risk_type,
                    order=cfg.training.order,
                )
                
                monitor = MonitorMV(SAVE_DIR)
                optimizer = Adam(pre_train_model.parameters(), lr=cfg.training.lr)

                lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

                pre_train_model, _, best_train_stats, train_error, test_error, time = stochastic_routine(pretrainloader, testloader, pre_train_model, optimizer, bound_prior, cfg.bound.type, loss=loss, monitor=monitor, num_epochs=10, lr_scheduler=lr_scheduler, model_type=cfg.training.risk, order = cfg.training.order, pretraining = True)
        
                posts = pre_train_model.get_post()
                monitor.close()

                betas = [p.detach() for p in posts]

                model = MultipleMajorityVote(
                    predictors,
                    betas,
                    weights=(0.5, 0.5),
                    mc_draws=cfg.training.MC_draws,
                    distr=distr,
                    kl_factor=kl_factor,
                    risk_type=cfg.training.risk_type,
                    order=cfg.training.order,
                )
                trainloader = [
                    DataLoader(train1, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True),
                    DataLoader(train2, batch_size=cfg.training.batch_size // 2, num_workers=cfg.num_workers, shuffle=True)
                ]

            
            monitor = MonitorMV(SAVE_DIR)
            optimizer = Adam(model.parameters(), lr=cfg.training.lr)
            # init learning rate scheduler
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

            best_model, _, best_train_stats, train_error, test_error, time = stochastic_routine(trainloader, testloader, model, optimizer, bound, cfg.bound.type, loss=loss, monitor=monitor, num_epochs=cfg.training.num_epochs, lr_scheduler=lr_scheduler, model_type=cfg.training.risk, order = cfg.training.order)
            best_thetas = best_model.best_thetas

            if isinstance(best_model, MultipleMajorityVote):
                # We compare two data-dependent candidates.  Allocating delta/2
                # to each individual bound preserves an overall confidence of
                # at least 1-delta after selecting the smaller one.
                individual_delta = cfg.bound.delta / len(best_model.mvs)
                individual_bound = lambda n, individual_model, risk: BOUNDS[cfg.bound.type](
                    n,
                    individual_model,
                    risk,
                    delta=individual_delta,
                    coeff=coeff,
                    order=cfg.training.order,
                    div=div,
                    type=cfg.training.risk,
                )
                selected_model, selected_index, individual_stats, selected_test_stats = select_best_individual_vote(
                    trainloader,
                    testloader,
                    best_model,
                    individual_bound,
                    cfg.bound.type,
                    loss=loss,
                    best_posts=best_thetas,
                )

                seed_results["cross-train-error"] = train_error["error"]
                seed_results["cross-test-error"] = test_error["error"]
                seed_results[f"cross-{cfg.bound.type}"] = best_train_stats[cfg.bound.type]
                seed_results["individual-bounds"] = [stats[cfg.bound.type] for stats in individual_stats]
                seed_results["individual-train-errors"] = [stats["error"] for stats in individual_stats]
                seed_results["selected-vote"] = selected_index
                seed_results["selection-delta"] = individual_delta
                seed_results["theta"] = best_thetas[selected_index].detach()

                selected_train_stats = individual_stats[selected_index]
                seed_results["train-error"] = selected_train_stats["error"]
                seed_results["test-error"] = selected_test_stats["error"]
                seed_results["train-risk"] = selected_train_stats["error"]
                seed_results[cfg.bound.type] = selected_train_stats[cfg.bound.type]
                seed_results["posterior"] = selected_model.get_post().detach()
                seed_results["strength"] = selected_train_stats["strength"]
                seed_results["KL"] = selected_model.KL().item()
                seed_results["entropy"] = selected_model.entropy().item()
                print(
                    f"Selected vote {selected_index}: {cfg.bound.type}="
                    f"{selected_train_stats[cfg.bound.type]}; "
                    f"test error={selected_test_stats['error']}"
                )
            else:
                seed_results["train-error"] = train_error["error"]
                seed_results["test-error"] = test_error["error"]
                seed_results["train-risk"] = best_train_stats["error"]
                seed_results[cfg.bound.type] = best_train_stats[cfg.bound.type]
                seed_results["posterior"] = best_model.get_post().detach()
                seed_results["theta"] = best_thetas.detach()
                seed_results["strength"] = best_train_stats["strength"]
                seed_results["KL"] = best_model.KL().item()
                seed_results["entropy"] = best_model.entropy().item()

            seed_results["time"] = time

            # save seed results
            np.save(result_file, seed_results)
            monitor.close()

        train_errors.append(seed_results["train-error"])
        test_errors.append(seed_results["test-error"])
        entropies.append(seed_results["entropy"])
        strengths.append(seed_results["strength"])
        kls.append(seed_results["KL"])
        bounds.append(seed_results[cfg.bound.type])
        times.append(seed_results["time"])
        train_losses.append(seed_results.pop("train-risk", None)) # available only for non-exact methods
        if "selected-vote" in seed_results:
            selected_votes.append(seed_results["selected-vote"])
            individual_bounds.append(seed_results["individual-bounds"])
            cross_bounds.append(seed_results[f"cross-{cfg.bound.type}"])
            cross_test_errors.append(seed_results["cross-test-error"])
 
    results = {"train-error": (np.mean(train_errors), np.std(train_errors)),"test-error": (np.mean(test_errors), np.std(test_errors)), cfg.bound.type: (np.mean(bounds), np.std(bounds)), "time": (np.mean(times), np.std(times)), "strength": (np.mean(strengths), np.std(strengths)), "train-risk": (np.mean(train_losses), np.std(train_losses)), "entropy": (np.mean(entropies), np.std(entropies)), "KL": (np.mean(kls), np.std(kls))}

    if selected_votes:
        results["selection-rate-vote-0"] = np.mean(np.asarray(selected_votes) == 0)
        results["individual-bounds"] = (np.mean(individual_bounds, axis=0), np.std(individual_bounds, axis=0))
        results[f"cross-{cfg.bound.type}"] = (np.mean(cross_bounds), np.std(cross_bounds))
        results["cross-test-error"] = (np.mean(cross_test_errors), np.std(cross_test_errors))

    np.save(ROOT_DIR / "err-b.npy", results)

if __name__ == "__main__":
    main()
