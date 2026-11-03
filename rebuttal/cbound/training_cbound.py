"""C-bound on the binary datasets of the paper.

The objective and the certificate are those of the joint algorithm of Viallard et al.
(2021, Algorithm 3), from their code in ECML21-PB-CBound/, unmodified (commit fea7674).
The data, the voters and the optimization protocol are those of the other methods of
the paper: Adam, shuffled mini-batches, learning-rate decay, early stopping and best
epoch according to the bound, random initialization.
"""
import logging
import random
import sys
from pathlib import Path
from time import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# both repositories have a core/ package: theirs is the one used here
sys.path[:0] = [str(HERE / "ECML21-PB-CBound"), str(ROOT)]

import cvxpy as cp
import hydra
import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from core.metrics import Metrics
from data.datasets import Dataset
from learner.c_bound_joint_learner import CBoundJointLearner
from models.stumps import uniform_decision_stumps
from voter.majority_vote import MajorityVote
from voter.majority_vote_diff import MajorityVoteDiff


VACUOUS = 0.999


class PaperStumpsMV(MajorityVote):
    """Their majority vote with the decision stumps of models/stumps.py and a uniform prior."""

    def __init__(self, X, y, nb_per_attribute):
        super().__init__(X, y, complemented=True, quasi_uniform=False)
        self.nb_per_attribute = nb_per_attribute
        self.fit()

    def fit(self):
        super().fit()
        X, _ = self.X_y_list
        self.stumps, M = uniform_decision_stumps(self.nb_per_attribute, X.shape[1], X.min(0), X.max(0), "categorical")
        self.voter_list = list(range(M))
        self.prior = np.ones((M, 1)) / M
        self.post = np.array(self.prior)
        return self

    def output(self, X):
        return self.stumps(torch.as_tensor(X))


def predict(vote, X, batch_size=4096):
    post = torch.as_tensor(vote.post, dtype=torch.float64)
    with torch.no_grad():
        return torch.cat([vote.output(x).double() @ post for x in torch.as_tensor(X).split(batch_size)]).numpy()


def run_seed(cfg, seed, data_path):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    data = Dataset(cfg.dataset, normalize=True, data_path=data_path, valid_size=0)
    X_train, X_test = data.X_train.astype(np.float32), data.X_test.astype(np.float32)
    y_train, y_test = data.y_train[:, None], data.y_test[:, None]
    n, delta = len(X_train), cfg.bound.delta

    vote = PaperStumpsMV(X_train, y_train, cfg.model.M)
    learner = CBoundJointLearner(vote, delta=delta, t=cfg.training.barrier)
    learner.mv_diff = MajorityVoteDiff(vote, learner.device)
    # same initialization as stated in the paper for FO, SO and Bin
    learner.mv_diff._post.data = torch.log(0.01 + 1.99 * torch.rand_like(learner.mv_diff._post.data))

    optimizer = Adam(learner.mv_diff.parameters(), lr=cfg.training.lr)
    learner._optim = optimizer  # used by their _optimize instead of COCOB
    learner._log = {}
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    loader = DataLoader(train, batch_size=cfg.training.batch_size, shuffle=True)

    best_bound, no_improv, best_post, best_epoch, skipped = float("inf"), 0, None, -1, 0
    certificate = Metrics("CBoundJoint", vote, delta=delta).fit
    t1 = time()
    for e in range(cfg.training.num_epochs):
        for x, y in loader:
            # their learner requires both classes in a batch and a solution of the inner problem
            if len(torch.unique(y)) < 2:
                skipped += 1
                continue
            try:
                learner._optimize({"x": x, "y": y, "m": n})
            except (TypeError, cp.error.SolverError):
                skipped += 1

        vote.post = torch.softmax(learner.mv_diff._post.detach(), 0).numpy()
        bound = float(certificate(y_train, predict(vote, X_train)))

        if bound < best_bound:
            best_bound, no_improv, best_post, best_epoch = bound, 0, vote.post.copy(), e
        elif bound < VACUOUS:
            no_improv += 1

        # the C-bound is capped at 1, unlike the other bounds: a vacuous epoch carries no signal
        if bound < VACUOUS:
            lr_scheduler.step(bound)
        if no_improv >= max(1, cfg.training.num_epochs // 4):
            break
    elapsed = time() - t1

    vote.post = best_post
    y_p_train, y_p_test = predict(vote, X_train), predict(vote, X_test)
    return {
        "joint": best_bound,
        "seeger": float(Metrics("CBoundSeeger", vote, delta=delta).fit(y_train, y_p_train)),
        # the vote errs when at least half of the weight is wrong, as for the other methods
        "train-error": float(np.mean(y_train * y_p_train <= 0)),
        "test-error": float(np.mean(y_test * y_p_test <= 0)),
        "gibbs-risk": float(Metrics("Risk").fit(y_train, y_p_train)),
        "disagreement": float(Metrics("Disagreement").fit(y_train, y_p_train)),
        "KL": float(np.sum(vote.post * np.log(vote.post / vote.prior))),
        "time": elapsed,
        "best-epoch": best_epoch,
        "skipped-batches": skipped,
        "posterior": vote.post,
    }


@hydra.main(config_path="config/cbound.yaml")
def main(cfg):
    logging.getLogger().setLevel(logging.WARNING)
    root = HERE / "results" / cfg.dataset
    keys = ["joint", "seeger", "test-error", "train-error", "gibbs-risk", "disagreement", "KL", "time", "best-epoch", "skipped-batches"]

    all_results = []
    for i in range(cfg.training.first_trial, cfg.training.first_trial + cfg.training.num_trials):
        seed = cfg.training.seed + i
        save_dir = root / f"seed={seed}"
        result_file = save_dir / "err-b.npy"
        if result_file.is_file():
            seed_results = np.load(result_file, allow_pickle=True).item()
        else:
            seed_results = run_seed(cfg, seed, ROOT / "data")
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(result_file, seed_results)
        print(f"seed {seed}: joint={seed_results['joint']:.4f} seeger={seed_results['seeger']:.4f} test={seed_results['test-error']:.4f} KL={seed_results['KL']:.2f} epoch={seed_results['best-epoch']} skipped={seed_results['skipped-batches']} time={seed_results['time']:.1f}", flush=True)
        all_results.append(seed_results)

    if cfg.training.first_trial == 0 and cfg.training.num_trials == 10:
        np.save(root / "err-b.npy", {k: (np.mean([r[k] for r in all_results]), np.std([r[k] for r in all_results])) for k in keys})


if __name__ == "__main__":
    main()
