# RulesRepair

Repairing Decision Rules in Data-Aware Process Models

## Requirements

To run RulesRepair, you need to have installed:

* [Python 3.9](https://www.python.org/downloads/)
* [Poetry 2.2.1](https://python-poetry.org/docs/#installation), used to install the dependencies specified in `pyproject.toml`.
* [Graphviz](https://graphviz.org/download/), required by the `graphviz` Python package to render decision trees.
* [OpenJDK 8 or later](https://adoptium.net/) (optional), required only for the Weka C4.5/REPTree baselines.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/KDMG/RulesRepair/
cd RulesRepair/

poetry env use python3.9
poetry install
```

All commands below are executed through Poetry, so activating the virtual environment is not required.

## Reproducing the Experiments

To reproduce the experiments, first run the experimental pipeline for the desired dataset and then run the quantitative evaluation. The qualitative example reported in the paper can be reproduced separately as described below.

### Running the Experimental Pipeline

To run the experiments for a dataset, execute:

```bash
poetry run python run_pipeline.py --dataset <dataset> --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

Replace `<dataset>` with the desired dataset name, e.g., `sepsis`. The available datasets are listed in the [datasets](https://github.com/KDMG/RulesRepair/tree/main/datasets) folder.

### Output Layout

All files produced by `run_pipeline.py` for a dataset are stored under:

```text
experiments/<dataset>/
```

The directory contains:

* `<dataset>_cut/` — the mined Petri net (`pn_normative.pnml`) and the normative, training, and test XES splits;
* `decision_points/` — the extracted decision-point tables and `normative_model.pkl`, used to generate the mutations;
* `repair/` — the repair results, organized by seed.

If the required files already exist in `<dataset>_cut/` and `decision_points/`, `run_pipeline.py` reuses them instead of overwriting them. This also allows the pipeline to be executed with custom inputs. For example, a specific Petri net for a dataset can be placed directly in the corresponding `<dataset>_cut/` directory.

## Quantitative Evaluation

The following commands reproduce the quantitative analyses reported in the paper.

### Computational Time

```bash
poetry run python -m analysis.core.computational_time \
    experiments/<dataset>/repair/seed_*/*/mutated/results.csv
```

The command reports the mean, median, and standard deviation of the training time for RulesRepair and each baseline. Results are provided overall, by mutation type, and by decision point, and are saved under:

```text
evaluation/quantitative/timing/
```

### RQ1 — Dominance over the Baselines

Run:

```bash
poetry run python -m analysis.rq1 cart-summary \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    -out-csv evaluation/quantitative/rq1/summary_<dataset>.csv
```

and:

```bash
poetry run python -m analysis.rq1 dominance-advantage \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --group-summary-out evaluation/quantitative/rq1/group_summary_<dataset>.csv \
    --mutation-summary-out evaluation/quantitative/rq1/mutation_summary_<dataset>.csv
```

Both commands report results overall and broken down by decision point and mutation type.

### RQ2 — Statistical Significance against the Baselines

Run:

```bash
poetry run python -m analysis.rq2.statistical rq-table \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --dataset <dataset> \
    --trial-level \
    --out-dir evaluation/quantitative/rq2
```

The command generates one table per decision point, containing an `overall` row and one row for each mutation type.

To obtain results pooled into one row per dataset, add the `--by-dataset` option.

## Qualitative Evaluation

The Pareto explorer can be launched with:

```bash
poetry run python pareto_explorer/app.py
```

To reproduce the qualitative example reported in Figure 3 of the paper, open the following files from `evaluation/qualitative/` on the first screen:

* **Petri net:** `pn_normative_sepsis.pnml`
* **Model:** `normative_model_sepsis_p42_change_feature_example.pkl`
* **Log:** `sepsis_train.xes`

Then select decision point **p_42**.

This reproduces the example reported in the paper, with the following values:

* **Normative:** Accuracy 0.728, Simplicity 0.710
* **Repaired:** Accuracy 0.968, Simplicity 0.710, Similarity 0.667
* **CART:** Accuracy 0.956, Simplicity 0.645, Similarity 0.000

These values can be verified against:

```text
experiments/sepsis/repair/seed_3/p_42/mutated/results.csv
```

using trial `s3_change_feature_0`.

### Reproducibility Note

The provided `normative_model_sepsis_p42_change_feature_example.pkl` is identical to:

```text
experiments/sepsis/decision_points/normative_model.pkl
```

except for the tree associated with decision point `p_42`. Its root split is set to:

```text
DiagnosticArtAstrup_True <= 0.0
```

corresponding to the specific `change_feature` mutation used in the paper's qualitative example.

Since Petri-net discovery is not fixed by the experimental seeds, a fresh execution of `run_pipeline.py` may produce a different root split for the same decision point. The provided model therefore fixes the specific mutated tree used in Figure 3, ensuring that the example can be reproduced exactly.

The model can be regenerated with:

```bash
poetry run python -c "
import pickle
from mutations.tree_mutations import apply_change_feature

with open('experiments/sepsis/decision_points/normative_model.pkl', 'rb') as f:
    model = pickle.load(f)

columns = model['p_42']['columns']
pos = columns.index('DiagnosticArtAstrup_True')
apply_change_feature(model['p_42']['tree'], pos, 0.0, columns)

with open(
    'evaluation/qualitative/normative_model_sepsis_p42_change_feature_example.pkl',
    'wb'
) as f:
    pickle.dump(model, f)
"
```

## Contact

For questions or further information, please contact:

| Contributor | Contact |
| :--- | :--- |
| Chiara Gobbi | c.gobbi@pm.univpm.it |
