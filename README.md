# RulesRepair

Repairing Decision Rules in Data-Aware Process Models 

## Requirements

To run our application you need to have installed:

* [Python 3.9](https://www.python.org/downloads/)

* [Poetry 2.2.1](https://python-poetry.org/docs/#installation) (the committed `poetry.lock` was generated with 2.2.1), to install the dependencies listed in `pyproject.toml`

* [Graphviz](https://graphviz.org/download/), required by the `graphviz` Python package to render trees

* [OpenJDK 8 or later](https://adoptium.net/) optional, only for the Weka C4.5/REPTree

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/KDMG/RulesRepair/
cd RulesRepair/

poetry env use python3.9
# poetry lock only if you have problems with the pre-generated lock file
poetry install
source $(poetry env info --path)/bin/activate
```

## Reproducing the experiments

For reproducing the experiments according to our experimental setup, first launch the repair algorithm for each dataset, then run the quantitative and qualitative evaluation.

### Experiments
To reproduce the experiments for a dataset, run:

```bash
poetry run python run_pipeline.py --dataset sepsis --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

Replace `sepsis` with the desired dataset name to reproduce the experiments for a different dataset.

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
you can find the files used to show the qualitative evaluation in the `evaluation/qualitative/` folder.

## Contact

For any information, please contact:

| Contributor name | Contacts |
| :-------- | :------- |
| `Chiara Gobbi` | c.gobbi@pm.univpm.it |
