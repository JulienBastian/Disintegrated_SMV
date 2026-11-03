from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributions import Dirichlet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from core.bounds import BOUNDS
from core.losses import moment_loss, rand_loss, sigmoid_loss
from core.utils import deterministic
from data.datasets import Dataset, TorchDataset
from models.majority_vote import MajorityVote, MultipleMajorityVote
from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps
from optimization import evaluate, select_best_individual_vote, stochastic_routine


def method_settings(cfg):
    settings = {
        "exact": (None, 1.0, "dirichlet", 1.0),
        "MC": (
            lambda x, y, z: sigmoid_loss(x, y, z, c=cfg.training.sigmoid_c),
            1.0,
            "dirichlet",
            1.0,
        ),
        "Rnd": (
            lambda x, y, z: rand_loss(x, y, z, n=cfg.training.rand_n),
            2.0,
            "categorical",
            cfg.training.rand_n,
        ),
        "FO": (
            lambda x, y, z: moment_loss(x, y, z, order=1),
            2.0,
            "categorical",
            1.0,
        ),
        "SO": (
            lambda x, y, z: moment_loss(x, y, z, order=2),
            4.0,
            "categorical",
            2.0,
        ),
        "des_dir": (
            lambda x, y, z: sigmoid_loss(x, y, z, c=cfg.training.sigmoid_c),
            1.0,
            "dirichlet",
            1.0,
        ),
        "des_rivasplata_dir": (
            lambda x, y, z: sigmoid_loss(x, y, z, c=cfg.training.sigmoid_c),
            1.0,
            "dirichlet",
            1.0,
        ),
    }
    return settings[cfg.training.risk]


def divergence_type(cfg):
    return "Renyi" if cfg.training.risk == "des_dir" else "KL"


def output_directory(cfg):
    root = Path(hydra.utils.get_original_cwd()) / cfg.variability.output_root
    root /= cfg.dataset
    root /= cfg.training.risk
    root /= cfg.bound.type
    root /= f"optimize-bound={cfg.training.opt_bound}"
    root /= f"stochastic-bound={cfg.bound.stochastic}"
    root /= cfg.model.pred
    root /= f"learnt_prior={cfg.pretraining.learnt_prior}"
    root /= f"M={cfg.model.M}"
    root /= f"max-depth={cfg.model.tree_depth}"
    root /= f"prior={cfg.model.prior}"

    if cfg.model.uniform:
        root /= "uniform"
    else:
        root /= f"lr={cfg.training.lr}"
        root /= f"batch-size={cfg.training.batch_size}"

    if cfg.training.risk == "MC":
        root /= f"MC={cfg.training.MC_draws}"
    elif cfg.training.risk == "des_dir":
        root /= f"order={cfg.training.order}"
        root /= f"MC={cfg.training.MC_draws}"
        root /= f"risk-type={cfg.training.risk_type}"
    elif cfg.training.risk == "des_rivasplata_dir":
        root /= f"MC={cfg.training.MC_draws}"
        root /= f"risk-type={cfg.training.risk_type}"

    return root / f"seed={cfg.training.seed}"


def make_bound(cfg, coeff, div, delta, fixed_n=None):
    def bound(n, model, risk):
        sample_size = n if fixed_n is None else fixed_n
        return BOUNDS[cfg.bound.type](
            sample_size,
            model,
            risk,
            delta=delta,
            coeff=coeff,
            order=cfg.training.order,
            div=div,
            type=cfg.training.risk,
        )

    return bound


def make_model(cfg, predictors, priors, distr, kl_factor):
    common = {
        "mc_draws": cfg.training.MC_draws,
        "distr": distr,
        "kl_factor": kl_factor,
        "risk_type": cfg.training.risk_type,
        "order": cfg.training.order,
    }
    if cfg.model.pred == "rf":
        return MultipleMajorityVote(
            predictors,
            priors,
            weights=(0.5, 0.5),
            **common,
        )
    return MajorityVote(predictors, priors, **common)


def train_once(cfg, loss, coeff, distr, kl_factor, div):
    deterministic(cfg.training.seed)
    data = Dataset(
        cfg.dataset,
        normalize=True,
        data_path=Path(hydra.utils.get_original_cwd()) / "data",
        valid_size=0,
    )

    if cfg.model.pred == "stumps-uniform":
        predictors, num_voters = uniform_decision_stumps(
            cfg.model.M,
            data.X_train.shape[1],
            data.X_train.min(0),
            data.X_train.max(0),
            distr,
        )
        trainloader = DataLoader(
            TorchDataset(data.X_train, data.y_train),
            batch_size=cfg.training.batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
        )
        priors = torch.ones(num_voters) * cfg.model.prior

    elif cfg.model.pred == "rf":
        tree_depth = cfg.model.tree_depth
        if tree_depth == "None":
            tree_depth = None
        predictors, num_voters = two_forests(
            cfg.model.M,
            0.5,
            data.X_train,
            data.y_train,
            max_samples=cfg.model.bootstrap,
            max_depth=tree_depth,
            binary=data.binary,
        )
        split = len(data.X_train) // 2
        train1 = TorchDataset(data.X_train[split:], data.y_train[split:])
        train2 = TorchDataset(data.X_train[:split], data.y_train[:split])
        trainloader = [
            DataLoader(
                train1,
                batch_size=cfg.training.batch_size // 2,
                num_workers=cfg.num_workers,
                shuffle=True,
            ),
            DataLoader(
                train2,
                batch_size=cfg.training.batch_size // 2,
                num_workers=cfg.num_workers,
                shuffle=True,
            ),
        ]
        priors = [torch.ones(num_voters) * cfg.model.prior for _ in predictors]

        if cfg.pretraining.learnt_prior:
            pretrainloader = [
                DataLoader(
                    train2,
                    batch_size=cfg.training.batch_size // 2,
                    num_workers=cfg.num_workers,
                    shuffle=True,
                ),
                DataLoader(
                    train1,
                    batch_size=cfg.training.batch_size // 2,
                    num_workers=cfg.num_workers,
                    shuffle=True,
                ),
            ]
            prior_model = make_model(
                cfg,
                predictors,
                priors,
                distr,
                kl_factor,
            )
            prior_optimizer = Adam(prior_model.parameters(), lr=cfg.training.lr)
            prior_scheduler = ReduceLROnPlateau(
                prior_optimizer,
                "min",
                factor=0.1,
                patience=2,
            )
            prior_model, *_ = stochastic_routine(
                pretrainloader,
                DataLoader(
                    TorchDataset(data.X_test, data.y_test),
                    batch_size=4096,
                    num_workers=cfg.num_workers,
                    shuffle=False,
                ),
                prior_model,
                prior_optimizer,
                None,
                cfg.bound.type,
                loss=loss,
                num_epochs=10,
                lr_scheduler=prior_scheduler,
                model_type=cfg.training.risk,
                order=cfg.training.order,
                pretraining=True,
            )
            priors = [post.detach() for post in prior_model.get_post()]

    else:
        raise NotImplementedError(
            "model.pred must be either 'stumps-uniform' or 'rf'"
        )

    testloader = DataLoader(
        TorchDataset(data.X_test, data.y_test),
        batch_size=4096,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    model = make_model(cfg, predictors, priors, distr, kl_factor)

    training_bound = None
    if cfg.training.opt_bound:
        fixed_n = None if cfg.bound.stochastic else len(data.X_train)
        training_bound = make_bound(
            cfg,
            coeff,
            div,
            cfg.bound.delta,
            fixed_n=fixed_n,
        )

    optimizer = Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=2)
    best_model, _, best_stats, _, _, training_time = stochastic_routine(
        trainloader,
        testloader,
        model,
        optimizer,
        training_bound,
        cfg.bound.type,
        loss=loss,
        num_epochs=cfg.training.num_epochs,
        lr_scheduler=scheduler,
        model_type=cfg.training.risk,
        order=cfg.training.order,
    )
    return best_model, trainloader, testloader, best_stats, training_time


def select_deployment_vote(cfg, model, trainloader, testloader, loss, coeff, div):
    if not isinstance(model, MultipleMajorityVote):
        bound = make_bound(cfg, coeff, div, cfg.bound.delta)
        return model, trainloader, bound, None, None

    individual_delta = cfg.bound.delta / len(model.mvs)
    individual_bound = make_bound(cfg, coeff, div, individual_delta)
    selected_model, selected_index, individual_stats, _ = select_best_individual_vote(
        trainloader,
        testloader,
        model,
        individual_bound,
        cfg.bound.type,
        loss=loss,
        best_posts=model.best_thetas,
    )
    return (
        selected_model,
        trainloader[selected_index],
        individual_bound,
        selected_index,
        individual_stats,
    )


def draw_theta(model):
    if model.distr_type == "dirichlet":
        return Dirichlet(model.get_post().detach()).sample((1,))
    return model.get_theta(draw=True).detach()


def mean_theta(model):
    if model.distr_type == "dirichlet":
        alpha = model.get_post().detach()
        return (alpha / alpha.sum()).unsqueeze(0)
    return model.get_theta(draw=False).detach()


def evaluate_theta(dataloader, model, theta, bound=None, bound_type=None, loss=None):
    original_risk_type = model.risk_type
    try:
        # A final Dirichlet draw defines a deterministic majority vote even when
        # the hyper-posterior was trained with the stochastic-risk objective.
        if model.distr_type == "dirichlet":
            model.risk_type = "draw"
            model.distribution.thetas = theta
        bounds = None if bound is None else {bound_type: bound}
        return evaluate(
            dataloader,
            model,
            bounds=bounds,
            loss=loss,
            best_post=theta,
        )
    finally:
        model.risk_type = original_risk_type


def sample_variability(
    cfg,
    model,
    trainloader,
    testloader,
    bound,
    loss,
):
    test_errors = []
    bound_values = []
    sampled_thetas = []

    with torch.no_grad():
        for _ in range(cfg.variability.num_draws):
            theta = draw_theta(model)
            train_stats = evaluate_theta(
                trainloader,
                model,
                theta,
                bound=bound,
                bound_type=cfg.bound.type,
                loss=loss,
            )
            test_stats = evaluate_theta(testloader, model, theta)
            sampled_thetas.append(theta.cpu().numpy())
            bound_values.append(train_stats[cfg.bound.type])
            test_errors.append(test_stats["error"])

        fixed_theta = mean_theta(model)
        mean_train_stats = evaluate_theta(
            trainloader,
            model,
            fixed_theta,
            bound=bound,
            bound_type=cfg.bound.type,
            loss=loss,
        )
        mean_test_stats = evaluate_theta(testloader, model, fixed_theta)

    test_errors = np.asarray(test_errors, dtype=float)
    bound_values = np.asarray(bound_values, dtype=float)
    ddof = 1 if cfg.variability.num_draws > 1 else 0

    return {
        "num-draws": cfg.variability.num_draws,
        "hyperposterior-alpha": model.get_post().detach().cpu().numpy(),
        "draw-thetas": np.asarray(sampled_thetas),
        "draw-test-errors": test_errors,
        "draw-bounds": bound_values,
        "draw-test-error-mean": test_errors.mean(),
        "draw-test-error-variance": test_errors.var(ddof=ddof),
        "draw-test-error-std": test_errors.std(ddof=ddof),
        "draw-test-error-quantiles": np.quantile(test_errors, [0.05, 0.5, 0.95]),
        "draw-bound-mean": bound_values.mean(),
        "draw-bound-variance": bound_values.var(ddof=ddof),
        "draw-bound-std": bound_values.std(ddof=ddof),
        "draw-bound-quantiles": np.quantile(bound_values, [0.05, 0.5, 0.95]),
        "mean-vote-theta": fixed_theta.detach().cpu().numpy(),
        "mean-vote-test-error": mean_test_stats["error"],
        "mean-vote-bound-value": mean_train_stats[cfg.bound.type],
        "mean-vote-test-error-variance": 0.0,
        "mean-vote-bound-variance": 0.0,
    }


@hydra.main(config_path="config/variability.yaml")
def main(cfg):
    save_dir = output_directory(cfg)
    output_file = save_dir / "final-draw-variability.npy"
    if output_file.is_file() and not cfg.variability.overwrite:
        print(f"Skipping existing result: {output_file}")
        return

    loss, coeff, distr, kl_factor = method_settings(cfg)
    div = divergence_type(cfg)
    model, trainloader, testloader, training_stats, training_time = train_once(
        cfg,
        loss,
        coeff,
        distr,
        kl_factor,
        div,
    )
    selected_model, selected_trainloader, selected_bound, selected_index, individual_stats = select_deployment_vote(
        cfg,
        model,
        trainloader,
        testloader,
        loss,
        coeff,
        div,
    )
    results = sample_variability(
        cfg,
        selected_model,
        selected_trainloader,
        testloader,
        selected_bound,
        loss,
    )
    results.update(
        {
            "selected-vote": selected_index,
            "individual-selection-stats": individual_stats,
            "training-stats": training_stats,
            "training-time": training_time,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_file, results)
    print(f"Saved variability results to {output_file}")


if __name__ == "__main__":
    main()
