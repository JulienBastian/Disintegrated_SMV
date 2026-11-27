# On the Disintegration of the Stochastic Majority Vote

This repository contains the implementation of the experiments from the paper:

> **On the Disintegration of the Stochastic Majority Vote: From PAC-Bayesian Bounds to a Self-Bounding Algorithm**


## Running the experiments

```bash
python -m pip install -r requirements-analysis.txt
```

To launch all binary and multiclass experiments locally:

```bash
./training.sh
```


To run a single configuration, use the following syntax :

```bash
python training.py \
  dataset=MUSH \
  model.pred=stumps-uniform \
  model.M=10 \
  training.risk=des_rivasplata_dir \
  training.risk_type=draw \
  training.batch_size=128 \
  model.prior=0.1
```

Datasets are downloaded automatically into `data/`. Results are saved under `results/`, with execution logs under `logs/`.

For random-forest cross-bounding runs, the two individual bounds are evaluated
after training with confidence level `delta / 2` each.  The vote with the
smallest bound is retained: `test-error`, the configured bound, `posterior`, and
`theta` refer to that single vote.  The result file also records the zero-based
`selected-vote`, `individual-bounds`, `cross-test-error`, and the original
combined bound under `cross-<bound type>`. As in the paper, the analysis
notebook reports the retained vote for the disintegrated methods (DIS-R and
DIS-V) and the combination of the two votes (`cross-*` entries) for the other
methods.

## C-bound baseline

`training.risk=CB` learns the C-bound with the joint algorithm of Viallard et al.
(2021, Algorithm 3), whose code is included in `third_party/pb_cbound/` (MIT
license, only its imports were made relative). `cbound.py` applies it to the
voters, mini-batches and optimization protocol of the other methods (Adam,
learning-rate decay, early stopping and best epoch according to the bound). On
multiclass datasets, the C-bound is computed from the margin of the vote, as in
the thesis of Viallard (2022, Chapter 4); `scripts/check_cbound_margin.py` checks
this reduction. Its certificate is saved under the `cbound` key. As in the paper,
its runtime is not reported by the analysis notebook.

```bash
python training.py \
  dataset=HABER \
  model.pred=stumps-uniform \
  model.M=10 \
  training.risk=CB \
  training.batch_size=128 \
  model.prior=1
```

## Analysing the results

```bash
jupyter notebook Analysis/Analysis.ipynb
```

To run the separate final-draw variability pipeline and analyse its results:

```bash
./variability.sh
jupyter notebook Analysis/Variability.ipynb
```

The variability pipeline trains once per dataset/hyperparameter configuration,
selects the best cross-bounding vote once, and evaluates 100 draws from that
fixed learned hyper-posterior. Its results are written under
`results_variability/`, leaving the standard experiment outputs unchanged.
