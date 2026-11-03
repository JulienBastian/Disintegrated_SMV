"""C-bound baseline (training.risk=CB).

The objective and the certificate are those of the joint algorithm of Viallard et al.
(2021, Algorithm 3), i.e. PAC-Bound 2 of Germain et al. (2015), from their code in
third_party/pb_cbound/. The data, the voters and the optimization protocol are those
of the other methods: Adam, shuffled mini-batches, learning-rate decay, early stopping
and best epoch according to the bound, random initialization.

Multiclass majority votes are handled through their margin, as in the thesis of
Viallard (2022, Chapter 4): with m(x) = sum_h rho_h (2 I[h(x) = y] - 1), the Gibbs risk,
the disagreement and the joint error are 0.5 (1 - m), 0.5 (1 - m^2) and their
difference. The joint learner and certificate of Viallard et al. compute exactly these
quantities when the outputs of the voters are y' (2 I[h(x) = y] - 1) with labels
y' in {-1, +1}, which only changes the sign of the margin. As for the other methods,
the two forests are learned on complementary halves of the training set, each vote is
learned and certified on the half its trees were not trained on, with confidence
delta / 2, and the vote with the smallest bound is retained.
"""
from time import time

import cvxpy as cp
import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from models.random_forest import two_forests
from models.stumps import uniform_decision_stumps
from third_party.pb_cbound.core.metrics import Metrics
from third_party.pb_cbound.learner.c_bound_joint_learner import CBoundJointLearner
from third_party.pb_cbound.voter.majority_vote import MajorityVote
from third_party.pb_cbound.voter.majority_vote_diff import MajorityVoteDiff

# the C-bound is capped at 1, unlike the other bounds: a vacuous epoch carries no signal
VACUOUS = 0.999


class StumpsMV(MajorityVote):
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


class SignedCorrectnessMV(MajorityVote):
    """Their majority vote whose inputs are already the signed correctness of its voters."""

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


def joint_learner(vote, cfg, delta):
    learner = CBoundJointLearner(vote, delta=delta, t=cfg.training.barrier)
    learner.mv_diff = MajorityVoteDiff(vote, learner.device)
    # same initialization as the other methods
    learner.mv_diff._post.data = torch.log(0.01 + 1.99 * torch.rand_like(learner.mv_diff._post.data))
    optimizer = Adam(learner.mv_diff.parameters(), lr=cfg.training.lr)
    learner._optim = optimizer  # used by their _optimize instead of COCOB
    learner._log = {}
    return learner, optimizer


def learner_step(learner, x, y, m):
    """One step of their joint algorithm; False when it cannot be applied to the batch."""
    # their learner requires both classes in a batch and a solution of the inner problem
    if len(torch.unique(y)) < 2:
        return False
    try:
        learner._optimize({"x": x, "y": y, "m": m})
    except (TypeError, cp.error.SolverError):
        return False
    return True


def stats(vote, x, y):
    """Statistics of the other methods, computed from the outputs x of the voters."""
    post = vote.post[:, 0]
    return {
        "train-risk": float(Metrics("Risk").fit(y, x @ vote.post)),  # Gibbs risk
        "strength": float(np.mean(x * y > 0)),  # expected accuracy of a voter
        "KL": float(np.sum(post * np.log(post / vote.prior[:, 0]))),
        "entropy": float(-np.sum(post * np.log(post))),
    }


def binary_routine(cfg, data):
    X_train, X_test = data.X_train.astype(np.float32), data.X_test.astype(np.float32)
    y_train, y_test = data.y_train[:, None], data.y_test[:, None]
    n, delta = len(X_train), cfg.bound.delta

    vote = StumpsMV(X_train, y_train, cfg.model.M)
    learner, optimizer = joint_learner(vote, cfg, delta)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    loader = DataLoader(train, batch_size=cfg.training.batch_size, shuffle=True)

    def predict(X, batch_size=4096):
        post = torch.as_tensor(vote.post, dtype=torch.float64)
        with torch.no_grad():
            return torch.cat([vote.output(x).double() @ post for x in torch.as_tensor(X).split(batch_size)]).numpy()

    certificate = Metrics("CBoundJoint", vote, delta=delta).fit
    best_bound, no_improv, best_post, best_epoch, skipped = float("inf"), 0, None, -1, 0
    t1 = time()
    for e in range(cfg.training.num_epochs):
        for x, y in loader:
            skipped += not learner_step(learner, x, y, n)

        vote.post = torch.softmax(learner.mv_diff._post.detach(), 0).numpy()
        bound = float(certificate(y_train, predict(X_train)))

        if bound < best_bound:
            best_bound, no_improv, best_post, best_epoch = bound, 0, vote.post.copy(), e
        elif bound < VACUOUS:
            no_improv += 1

        if bound < VACUOUS:
            lr_scheduler.step(bound)
        if no_improv >= max(1, cfg.training.num_epochs // 4):
            break
    elapsed = time() - t1

    vote.post = best_post
    y_p_train, y_p_test = predict(X_train), predict(X_test)
    return {
        cfg.bound.type: best_bound,
        # the vote errs when at least half of the weight is wrong, as for the other methods
        "train-error": float(np.mean(y_train * y_p_train <= 0)),
        "test-error": float(np.mean(y_test * y_p_test <= 0)),
        "disagreement": float(Metrics("Disagreement").fit(y_train, y_p_train)),
        **stats(vote, vote.output(X_train).numpy(), y_train),
        "time": elapsed,
        "best-epoch": best_epoch,
        "skipped-batches": skipped,
        "posterior": vote.post,
    }


def multiclass_routine(cfg, data):
    delta = cfg.bound.delta / 2  # one certificate per vote

    tree_depth = None if cfg.model.tree_depth == "None" else cfg.model.tree_depth
    voters, M = two_forests(cfg.model.M, 0.5, data.X_train, data.y_train, max_samples=cfg.model.bootstrap, max_depth=tree_depth, binary=data.binary)
    m_train = len(data.X_train) // 2
    # each vote is learned and certified on the split its trees were not trained on
    splits = [(data.X_train[m_train:], data.y_train[m_train:]), (data.X_train[:m_train], data.y_train[:m_train])]

    votes, learners, schedulers, loaders, trains, tests = [], [], [], [], [], []
    for k, (X, y) in enumerate(splits):
        x_train, y_train = signed_correctness(voters[k], X, y)
        x_test, y_test = signed_correctness(voters[k], data.X_test, data.y_test)
        vote = SignedCorrectnessMV(M)
        learner, optimizer = joint_learner(vote, cfg, delta)

        votes.append(vote)
        learners.append(learner)
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
            for learner, (x, y), (x_train, _) in zip(learners, batches, trains):
                skipped += not learner_step(learner, x, y, len(x_train))

        for vote, learner in zip(votes, learners):
            vote.post = torch.softmax(learner.mv_diff._post.detach(), 0).numpy()
        bound = float(np.mean(certificates()))  # tau = 1/2

        if bound < best_bound:
            best_bound, no_improv, best_posts, best_epoch = bound, 0, [v.post.copy() for v in votes], e
        elif bound < VACUOUS:
            no_improv += 1

        if bound < VACUOUS:
            for scheduler in schedulers:
                scheduler.step(bound)
        if no_improv >= max(1, cfg.training.num_epochs // 4):
            break
    elapsed = time() - t1

    for vote, post in zip(votes, best_posts):
        vote.post = post
    bounds = certificates()
    # the vote errs when at least half of the weight is wrong, as for the other methods
    train_errors = [float(np.mean(x @ vote.post * y <= 0)) for vote, (x, y) in zip(votes, trains)]
    test_errors = [float(np.mean(x @ vote.post * y <= 0)) for vote, (x, y) in zip(votes, tests)]
    selected = int(np.argmin(bounds))
    x, y = trains[selected]

    return {
        cfg.bound.type: bounds[selected],
        "train-error": train_errors[selected],
        "test-error": test_errors[selected],
        "disagreement": float(Metrics("Disagreement").fit(y, x @ votes[selected].post)),
        **stats(votes[selected], x, y),
        "time": elapsed,
        "best-epoch": best_epoch,
        "skipped-batches": skipped,
        "selected-vote": selected,
        "selection-delta": delta,
        "individual-bounds": bounds,
        "individual-train-errors": train_errors,
        # bound on the tau = 1/2 average of the risks of the two votes
        f"cross-{cfg.bound.type}": float(np.mean(bounds)),
        "cross-train-error": float(np.mean(train_errors)),
        "cross-test-error": float(np.mean(test_errors)),
        "posterior": votes[selected].post,
    }


def cbound_routine(cfg, data):
    if cfg.model.pred == "stumps-uniform":
        return binary_routine(cfg, data)
    if cfg.model.pred == "rf":
        return multiclass_routine(cfg, data)
    raise NotImplementedError("model.pred should be one the following: [stumps-uniform, rf]")
