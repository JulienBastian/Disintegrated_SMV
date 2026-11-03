"""Checks that the binary metrics of Viallard et al., fed with signed correctness, give the multiclass quantities of the thesis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from third_party.pb_cbound.core.metrics import Metrics

rng = np.random.default_rng(0)
for _ in range(20):
    n, M, K = 3000, 50, 6
    y = rng.integers(0, K, n)
    votes = np.where(rng.random((n, M)) < 0.55, y[:, None], rng.integers(0, K, (n, M)))
    post = rng.dirichlet(np.ones(M))[:, None]
    margin = (2. * (votes == y[:, None]) - 1.) @ post  # thesis, Chapter 4
    signs = np.where(np.arange(n) % 2 == 0, 1, -1)[:, None]
    y_p = signs * margin
    risk, disa, joint = Metrics("Risk").fit(signs, y_p), Metrics("Disagreement").fit(signs, y_p), Metrics("Joint").fit(signs, y_p)
    w = ((votes != y[:, None]) * 1.) @ post
    assert np.isclose(risk, np.mean(0.5 * (1 - margin))) and np.isclose(risk, np.mean(w))
    assert np.isclose(disa, np.mean(0.5 * (1 - margin ** 2)))
    assert np.isclose(joint, np.mean(w ** 2))
    assert np.isclose(joint, risk - disa / 2)
    # the C-bound dominates the risk of the arg max vote
    scores = np.stack([((votes == k) * 1.) @ post[:, 0] for k in range(K)], 1)
    arg_max_risk = np.mean(scores.argmax(1) != y)
    c_bound = 1 - (1 - 2 * risk) ** 2 / (1 - 2 * disa)
    assert arg_max_risk <= np.mean(margin <= 0) <= c_bound + 1e-12
print("all checks passed")
