# RulesRepair

Repairing Decision Rules in Data-Aware Process Models 

## Requirements

To run our application you need to have installed:

* [Python 3.9](https://www.python.org/downloads/)

* [Poetry 2.2.1](https://python-poetry.org/docs/#installation) (the `poetry.lock` was generated with 2.2.1), to install the dependencies listed in `pyproject.toml`

* [Graphviz](https://graphviz.org/download/), required by the `graphviz` Python package to render trees

* [OpenJDK 8 or later](https://adoptium.net/) optional, only for the Weka C4.5/REPTree

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/KDMG/RulesRepair/
cd RulesRepair/

poetry env use python3.9
poetry install
# poetry lock if your received an error (you may have a poetry version different from 2.2.1)
source $(poetry env info --path)/bin/activate
```

## Reproducing the experiments

For reproducing the experiments according to our experimental setup, first launch the repair algorithm for each `dataset`, then run the quantitative and qualitative evaluation.

### Experiments
To reproduce the experiments for a dataset, run:

```bash
poetry run python run_pipeline.py --dataset <dataset> --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

Replace `<dataset>` with the desired dataset name, e.g., `<dataset>`= `sepsis` (see the [datasets](https://github.com/KDMG/RulesRepair/tree/main/datasets) folder).

### Output layout

Everything `run_pipeline.py` produces for a dataset lives under `experiments/<dataset>/`:

* `<dataset>_cut/` -- the mined Petri net (`pn_normative.pnml`) and the normative/train/test XES splits
* `decision_points/` -- the extracted decision point tables and `normative_model.pkl` to mutate
* `repair/` -- the repair results, per seed

If the files in `<dataset>_cut/` and `decision_points/` already exists, `run_pipeline.py` does NOT override.

### Quantitative evaluation

**Computational time:**
```bash
python -m analysis.core.computational_time experiments/<dataset>/repair/seed_*/*/mutated/results.csv
```
Prints and saves, under `evaluation/quantitative/timing/`, the mean/median/std training time per algorithm (RulesRepair and every baseline), broken down overall, by mutation type, and by decision point.

**RQ1 (dominance over the baselines):**
```bash
python -m analysis.rq1 cart-summary experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    -out-csv evaluation/quantitative/rq1/summary_<dataset>.csv

python -m analysis.rq1 dominance-advantage experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --group-summary-out evaluation/quantitative/rq1/group_summary_<dataset>.csv \
    --mutation-summary-out evaluation/quantitative/rq1/mutation_summary_<dataset>.csv
```
Both give results broken down by decision point and by mutation type as well as aggregated.

**RQ2 (statistical significance vs. the baselines):**
```bash
python -m analysis.rq2.statistical rq-table experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --dataset <dataset> --trial-level --out-dir evaluation/quantitative/rq2
```
Writes one table per decision point with an "overall" row plus one row per mutation type. Add `--by-dataset` for a version pooled into one row per dataset instead.

### Qualitative evaluation
You can inspect the Pareto explorer by running
```bash
python pareto_explorer/app.py
```
On screen 1, open the three files in `evaluation/qualitative/`:

* Petri net: `pn_normative_sepsis.pnml`
* Model: `normative_model_sepsis_p42_change_feature_example.pkl`
* Log: `sepsis_train.xes`

then click decision point **p_42**. This exactly reproduces the paper's Figure 3
example (Normative Acc 0.728/Simplicity 0.710, Repaired Acc 0.968/Simplicity
0.710/Similarity 0.667, CART Acc 0.956/Simplicity 0.645/Similarity 0.000, all
verified against `experiments/sepsis/repair/seed_3/p_42/mutated/results.csv`,
trial `s3_change_feature_0`). The model file is the same as the plain
`experiments/sepsis/decision_points/normative_model.pkl` except p_42's tree, whose root
split was set to `DiagnosticArtAstrup_True <= 0.0` -- the specific
`change_feature` mutation the paper's example starts from (Petri net mining
isn't seed-pinned, so a fresh `run_pipeline.py` run can mine a different root
split for the same decision-point name; this file freezes the one the figure
needs). Regenerate it with:
```bash
python -c "
import pickle, copy
from mutations.tree_mutations import apply_change_feature
with open('experiments/sepsis/decision_points/normative_model.pkl', 'rb') as f:
    model = pickle.load(f)
columns = model['p_42']['columns']
pos = columns.index('DiagnosticArtAstrup_True')
apply_change_feature(model['p_42']['tree'], pos, 0.0, columns)
with open('evaluation/qualitative/normative_model_sepsis_p42_change_feature_example.pkl', 'wb') as f:
    pickle.dump(model, f)
"
```

## Contact

For any information, please contact:

| Contributor name | Contacts |
| :-------- | :------- |
| `Chiara Gobbi` | c.gobbi@pm.univpm.it |
