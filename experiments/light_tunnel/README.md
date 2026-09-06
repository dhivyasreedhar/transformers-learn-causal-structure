# Light tunnel baseline: continuous CatFormer adaptation

This experiment runs the supplied CatFormer attention code on `lt_walks_v1/color_mix`, from **Causal chambers as a real-world physical testbed for AI methodology** (Gamella, Peters, Bühlmann, 2025). It tests prediction and exploratory dependency ranking. It does **not** reproduce the categorical, varying-transition task or staged-training theorem of **How Transformers Learn Causal Structure with Gradient Descent** (Nichani, Damian, Lee, 2024).

## Full-data rerun and reference-style figures

The newer `full_data.py` run uses all 10,000 color_mix rows: exhaustive row-target training, full validation and test evaluation, five blocked folds, and three seeds. Separate final models train on all rows for visualization. Run `plot_full_data.py` to generate the ground-truth / first-layer / second-layer / output-weight figures requested in the screenshot. `verify_full_data.py` audits complete coverage and split separation. See `full_data_report/REPORT.md` and `full_data_results/summary.json` for the newer results; the original results below describe the earlier small baseline.

## What is reused and changed

`vendor/catformer.py` is an unmodified copy of the supplied repository's model file (repository commit `7d5e4569f77485e7bbeb7ee7834fb56453439aa4`). `run.py` imports and subclasses `CatFormer`, reusing its causal softmax attention method directly. It retains two attention layers, concatenation of attention outputs into a growing residual stream, and direct trainable attention matrices. The first layer has three heads; the second has one.

Changes: continuous standardized values replace categorical tokens; variable identity and an observed/missing flag are embedded explicitly; a scalar linear readout and MSE replace categorical softmax and cross entropy; small random attention initialization breaks head symmetry; AdamW trains all weights jointly.

Each prompt consists of two complete support rows sampled from training data followed by one query row. Within each row variables are randomly ordered. The query target is always last and its value is zeroed with an explicit missing flag. Variable identity is retained. Training examples do not contain timestamps, intervention flags, graph labels, or test/validation support rows.

This is fixed-mechanism supervised masked regression with support examples. In-context mechanism learning is **not established** by this task: the network can memorize the shared physical mapping in its weights. The no-context ablation measures predictive performance when support values are removed.

## Two settings

- `rgb`: predict one of `ir_1`, `ir_2`, `ir_3`, `current` with only RGB values observed in the query. Other query sensor values are hidden. This is a prediction control with known candidate causes, not causal graph discovery.
- `all`: predict any of the seven variables from the other six. Evaluate query permutation reliance and first-layer query attention over all 42 ordered pairs, including 30 nonedges. These scores measure predictive dependence, not identified causal effects. Permutations can create off-distribution combinations.

The 12 ground-truth edges (RGB to every sensor/current output) are checked against the authors' standard light-tunnel graph and saved in `data/ground_truth.json`; they are used only for evaluation.

## Data and split

Selected columns: `red`, `green`, `blue`, `ir_1`, `ir_2`, `ir_3`, `current` (the tilde-I variables are `ir_*`, not `vis_*`). The original 10,000 measurements are retained. Train: rows 0–5899. Validation: 6100–7899. Test: 8100–9999. Rows 5900–6099 and 7900–8099 form 200-row gaps at the boundaries. Normalization is fit only on training rows. This single chronological split reduces adjacent-row leakage but is not proof of temporal independence.

All models are evaluated on the same 256 evenly spaced test rows per target. Ridge regression selects its penalty on validation data; the transformer selects its checkpoint on validation MSE. Three neural seeds are used for real data. Seed variation is not a statistical confidence interval over independent datasets. The fixed test prompts use only training rows for support.

Training-only Pearson correlations and quantile-binned chi-square divergence are exploratory signal diagnostics. The latter is ordinary pairwise dependence, sensitive to binning, **not** the conditional chi-square information appearing in the transformer paper's proof.

A separate synthetic control uses independent uniform RGB causes, a fixed dense linear 3-by-4 mixing matrix, and independent Gaussian sensor noise. It checks whether this adapter can learn an easy linear prediction problem.

## Reproduce

From this directory, create a Python environment and install the pinned dependencies:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python run.py --data data/color_mix.csv --out results --steps 1500 --seeds 0 1 2
.venv/bin/python run.py --data data/color_mix.csv --out synthetic_results --synthetic --modes rgb --steps 1500 --seeds 0
```

Results include JSON metrics, validation trajectories, NPZ weights, standardized predictions, and training normalization parameters. Convert standardized predictions back to physical measurement units using `summary.json`'s means and standard deviations.

## Sources and provenance

- Transformer paper: https://arxiv.org/abs/2402.14735
- Causal Chambers paper: https://doi.org/10.1038/s42256-024-00964-x
- Dataset documentation: https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_walks_v1
- Protocol: https://github.com/juangamella/causal-chamber/blob/main/datasets/lt_walks_v1/generators/color_mix.py
- Data ZIP: https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_walks_v1.zip
- Verified ZIP MD5: `dcad019186661a56de7a2d1db97fc2f0`

The dataset is licensed CC BY 4.0. Data credit: Juan L. Gamella, Jonas Peters, Peter Bühlmann. Raw `color_mix.csv` is unmodified; normalization and filtering to seven columns happen in memory. The generator drives RGB with deterministic waves, so “observational-only” describes the model's access to labels, not passive data collection.
