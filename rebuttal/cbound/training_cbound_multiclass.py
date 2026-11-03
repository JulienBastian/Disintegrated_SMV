"""C-bound on the multiclass datasets of the paper.

Multiclass majority votes are handled through their margin, as in the thesis of
Viallard (2023, Chapter 4): with m(x) = sum_h rho_h (2 I[h(x) = y] - 1), the Gibbs risk,
the disagreement and the joint error are 0.5 (1 - m), 0.5 (1 - m^2) and their
difference. The joint learner and certificate of ECML21-PB-CBound/ compute exactly
these quantities when the outputs of the voters are y' (2 I[h(x) = y] - 1) with labels
y' in {-1, +1}, which only changes the sign of the margin.

As for the other methods of the paper, the two forests are learned on complementary
halves of the training set, each vote is learned and certified on the half its trees
were not trained on, with confidence delta / 2, and the vote with the smallest bound
is retained. Optimization protocol: Adam, shuffled mini-batches, learning-rate decay,
early stopping and best epoch on the tau = 1/2 objective, random initialization.
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
from models.random_forest import two_forests
from voter.majority_vote import MajorityVote
from voter.majority_vote_diff import MajorityVoteDiff

VACUOUS = 0.999


class SignedCorrectnessMV(MajorityVote):
    """Majority vote whose inputs are already the signed correctness of its voters."""

    def __init__(self, M):
        super().__init__(np.zeros((1, 1)), np.ones((1, 1)), complemented=False, quasi_uniform=False)
        self.M = M
        self.fit()

    def fit(self):
        super().fit()
        self.voter_list = list(range(self.M))
        self.prior = np.ones((self.M, 1)) / self.M
        self.post = np.array(self.prior)
        return self

    def output(self, X):
        return torch.as_tensor(X)


def signed_correctness(voters, X, y):
    """Returns (y' * (2 I[h(x) = y] - 1), y') with alternating signs y'."""
    correct = 2. * (voters(X) == torch.as_tensor(y).float()[:, None]).double() - 1.
    signs = torch.where(torch.arange(len(y)) % 2 == 0, 1., -1.).double()[:, None]
    return (signs * correct).numpy(), signs.long().numpy()


def run_seed(cfg, seed, data_path):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    data = Dataset(cfg.dataset, normalize=True, data_path=data_path, valid_size=0)
    delta = cfg.bound.delta / 2  # one certificate per vote

    tree_depth = None if cfg.model.tree_depth == "None" else cfg.model.tree_depth
    voters, M = two_forests(cfg.model.M, 0.5, data.X_train, data.y_train, max_samples=cfg.model.bootstrap, max_depth=tree_depth, binary=data.binary)
    m_train = len(data.X_train) // 2
    # each vote is learned and certified on the split its trees were not trained on
    splits = [(data.X_train[m_train:], data.y_train[m_train:]), (data.X_train[:m_train], data.y_train[:m_train])]

    votes, learners, optimizers, schedulers, loaders, trains, tests = [], [], [], [], [], [], []
    for k, (X, y) in enumerate(splits):
        x_train, y_train = signed_correctness(voters[k], X, y)
        x_test, y_test = signed_correctness(voters[k], data.X_test, data.y_test)
        vote = SignedCorrectnessMV(M)
        learner = CBoundJointLearner(vote, delta=delta, t=cfg.training.barrier)
        learner.mv_diff = MajorityVoteDiff(vote, learner.device)
        # same initialization as stated in the paper for FO, SO and Bin
        learner.mv_diff._post.data = torch.log(0.01 + 1.99 * torch.rand_like(learner.mv_diff._post.data))
        optimizer = Adam(learner.mv_diff.parameters(), lr=cfg.training.lr)
        learner._optim = optimizer  # used by their _optimize instead of COCOB
        learner._log = {}

        votes.append(vote)
        learners.append(learner)
        optimizers.append(optimizer)
        schedulers.append(ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2))
        loaders.append(DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=cfg.training.batch_size // 2, shuffle=True))
        trains.append((x_train, y_train))
        tests.append((x_test, y_test))

    def certificates():
        return [float(Metrics("CBoundJoint", vote, delta=delta).fit(y, x @ vote.post)) for vote, (x, y) in zip(votes, trains)]

    best_bound, no_improv, best_posts, best_epoch, skipped = float("inf"), 0, None, -1, 0
    t1 = time()
    for e in range(cfg.training.num_epochs):
        for batches in zip(*loaders):
            for learner, (x, y), n in zip(learners, batches, [len(t[0]) for t in trains]):
                try:
                    learner._optimize({"x": x, "y": y, "m": n})
                except (TypeError, cp.error.SolverError):
                    skipped += 1

        for vote, learner in zip(votes, learners):
            vote.post = torch.softmax(learner.mv_diff._post.detach(), 0).numpy()
        bound = float(np.mean(certificates()))  # tau = 1/2

        if bound < best_bound:
            best_bound, no_improv, best_posts, best_epoch = bound, 0, [v.post.copy() for v in votes], e
        elif bound < VACUOUS:
            no_improv += 1

        # the C-bound is capped at 1, unlike the other bounds: a vacuous epoch carries no signal
        if bound < VACUOUS:
            for scheduler in schedulers:
                scheduler.step(bound)
        if no_improv >= max(1, cfg.training.num_epochs // 4):
            break
    elapsed = time() - t1

    for vote, post in zip(votes, best_posts):
        vote.post = post
    bounds = certificates()
    margins = [x @ vote.post * y for vote, (x, y) in zip(votes, tests)]
    # the vote errs when at least half of the weight is wrong, as for the other methods
    test_errors = [float(np.mean(margin <= 0)) for margin in margins]
    selected = int(np.argmin(bounds))
    x, y = trains[selected]
    y_p = x @ votes[selected].post

    return {
        "joint": bounds[selected],
        "test-error": test_errors[selected],
        "gibbs-risk": float(Metrics("Risk").fit(y, y_p)),
        "disagreement": float(Metrics("Disagreement").fit(y, y_p)),
        "KL": float(np.sum(votes[selected].post * np.log(votes[selected].post / votes[selected].prior))),
        "time": elapsed,
        "best-epoch": best_epoch,
        "skipped-batches": skipped,
        "selected-vote": selected,
        "selection-delta": delta,
        "individual-bounds": bounds,
        "individual-test-errors": test_errors,
        # bound on the tau = 1/2 average of the risks, as reported for the other methods
        "cross-joint": float(np.mean(bounds)),
        "cross-test-error": float(np.mean(test_errors)),
        "posterior": votes[selected].post,
    }


@hydra.main(config_path="config/cbound_multiclass.yaml")
def main(cfg):
    logging.getLogger().setLevel(logging.WARNING)
    root = HERE / "results" / cfg.dataset
    keys = ["joint", "test-error", "gibbs-risk", "disagreement", "KL", "time", "best-epoch", "skipped-batches", "cross-joint", "cross-test-error"]

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
        print(f"seed {seed}: joint={seed_results['joint']:.4f} cross={seed_results['cross-joint']:.4f} test={seed_results['test-error']:.4f} vote={seed_results['selected-vote']} KL={seed_results['KL']:.2f} epoch={seed_results['best-epoch']} skipped={seed_results['skipped-batches']} time={seed_results['time']:.1f}", flush=True)
        all_results.append(seed_results)

    if cfg.training.first_trial == 0 and cfg.training.num_trials == 10:
        np.save(root / "err-b.npy", {k: (np.mean([r[k] for r in all_results]), np.std([r[k] for r in all_results])) for k in keys})


if __name__ == "__main__":
    main()
