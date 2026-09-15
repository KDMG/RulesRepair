# RulesRepair

Repairing Decision Rules in Data-Aware Process Models

## Requirements

To run RulesRepair, you need to have installed:

* [Python 3.9](https://www.python.org/downloads/)
* [Poetry 2.2.1](https://python-poetry.org/docs/#installation), used to install the dependencies specified in `pyproject.toml`.
* [Graphviz](https://graphviz.org/download/), required by the `graphviz` Python package to render decision trees.
* [JDK 17](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html) (optional), required only for the Weka C4.5/REPTree.
  
## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/KDMG/RulesRepair/
cd RulesRepair/

poetry env use python3.9
poetry install
```

## Reproducing the Experiments

To reproduce the experiments according to our experimental setup, first launch the repair algorithm for each dataset, then run the quantitative and qualitative evaluation.

The experimental results and the evaluations reported in our paper are available in the [experiments](https://github.com/KDMG/RulesRepair/tree/main/experiments) [evaluation](https://github.com/KDMG/RulesRepair/tree/main/evaluation) and folders.

### Running the Experimental Pipeline

To run the experiments for a dataset, execute:

```bash
poetry run python run_pipeline.py --dataset <dataset> --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

Replace `<dataset>` with the desired dataset name, e.g., `sepsis`. The available datasets are listed in the [datasets](https://github.com/KDMG/RulesRepair/tree/main/datasets) folder.

#### Output Layout

All files produced by `run_pipeline.py` for a dataset are stored under `experiments/<dataset>/`, which contains:
* `<dataset>_cut/`: the mined Petri net (`pn_normative.pnml`) and the XES splits;
* `decision_points/`: the extracted decision points and `normative_model.pkl`, used to generate the mutations;
* `repair/`: the repair results, by seed.

If the required files already exist in `<dataset>_cut/` and `decision_points/`, `run_pipeline.py` reuses them instead of overwriting them. This also allows the pipeline to be executed with custom inputs. For example, a specific Petri net for a dataset can be placed directly in the corresponding `<dataset>_cut/` directory.

## Quantitative Evaluation

The following commands reproduce the quantitative analyses reported in the paper.

Every command below is run once per dataset. Most write their output under `evaluation/quantitative/<dataset>/...` by default; the exceptions are noted below.

### Computational Time

```bash
poetry run python -m analysis.core.computational_time \
    experiments/<dataset>/repair/seed_*/*/mutated/results.csv
```

For each trial, the command computes the difference between RulesRepair's total time (summed over its whole w_simp x w_simi grid for that trial) and the fastest baseline's ("Mine") single fit time (the minimum among CART/C4.5/REPTree). It reports the mean and standard deviation of this difference, aggregated by dataset only, and saves the result under:

```text
evaluation/quantitative/<dataset>/timing/time_delta_overall.csv
```

### RQ1 — Dominance over the Baselines

Run:

```bash
poetry run python -m analysis.rq1 dominance-summary \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv
```

and:

```bash
poetry run python -m analysis.rq1 dominance-advantage \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv
```

`dominance-summary` reports, for each Mine algorithm (CART/C4.5/REPTree), the percentage of times that at in which that baseline dominates at least one point of RulesRepair's Pareto front, aggregated by dataset, by decision point x mutation type, and by mutation type. `dominance-advantage` reports, for the trials where a baseline does dominate, the size of the accuracy/simplicity/similarity gap (median [Q1, Q3] of the per-trial median delta), aggregated the same three ways. Both save under `evaluation/quantitative/<dataset>/rq1/dominance_summary/` and `evaluation/quantitative/<dataset>/rq1/dominance_advantage/` respectively by default.

Once `dominance-summary` has been run for every dataset, combine the per-dataset `dominance_by_dataset.csv` files into the paper table:

```bash
poetry run python -m analysis.rq1 dominance-summary \
    -combine evaluation/quantitative/*/rq1/dominance_summary/dominance_by_dataset.csv
```

This writes the combined CSV and LaTeX table to `evaluation/quantitative/rq1/dominance_summary/dominance_frequency_table.csv`/`.tex`.

Once `dominance-advantage` has been run for every dataset, combine the per-dataset `dominance_advantage_by_dataset.csv` files into the paper table:

```bash
poetry run python -m analysis.rq1 dominance-advantage \
    -combine evaluation/quantitative/*/rq1/dominance_advantage/dominance_advantage_by_dataset.csv
```

This writes the combined CSV and LaTeX table to `evaluation/quantitative/rq1/dominance_advantage/dominance_magnitude_table.csv`/`.tex`.

### RQ2 — Statistical Significance against the Baselines

Run:

```bash
poetry run python -m analysis.rq2.statistical rq-table \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --dataset <dataset> \
    --trial-level
```

The command generates one table per decision point, containing an `overall` row and one row for each mutation type, saved under `evaluation/quantitative/<dataset>/rq2/rq_table/`. This compares RulesRepair against CART only.

To obtain results pooled into one row per dataset, add the `--by-dataset` option.

Once `rq-table` has been run for every dataset, combine the per-dataset CSVs into the paper table with `--combine`:

```bash
poetry run python -m analysis.rq2.statistical rq-table \
    --combine evaluation/quantitative/*/rq2/rq_table/rq_unified_trial_level_*.csv
```

For the paper table comparing RulesRepair against all three baselines (CART, C4.5, REPTree) at once, use `multi-baseline-table` instead, run once per dataset:

```bash
poetry run python -m analysis.rq2.statistical multi-baseline-table \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --dataset <dataset> \
    --trial-level --by-dataset
```

Saved under `evaluation/quantitative/<dataset>/rq2/multi_baseline_table/`.

Each dataset's `p_holm` at this point only corrects for that dataset/view's own test, not for the fact that CART, C4.5 and REPTree are three comparisons against the same cell. Once `multi-baseline-table` has been run for every dataset, get the final paper table by running `holm-table` on the per-dataset CSVs, which re-applies Holm-Bonferroni correction across the three baselines within each (dataset, view) group:

```bash
poetry run python -m analysis.rq2.statistical holm-table \
    evaluation/quantitative/*/rq2/multi_baseline_table/rq_unified_trial_level_by_dataset_*.csv
```

This writes the Hodges–Lehmann paper table (with the corrected significance markers) to `evaluation/quantitative/rq2/holm_table/rq_unified_trial_level_by_dataset_holm_corrected.csv`/`.tex`.

(`multi-baseline-table` also has its own `--combine` flag, which merges the per-dataset CSVs into the same table shape without the cross-baseline correction -- useful for a quick preview, but `holm-table` is the correct final step for the paper.)

### RQ2 — Attainment Figures

Run once per dataset to compute the attainment fields:

```bash
poetry run python -m analysis.rq2.attainment_figures outputs \
    experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --dataset <dataset>
```

Saved under `evaluation/quantitative/<dataset>/rq2/attainment_outputs/`.

To render the figures for one dataset, pooled across every decision point:

```bash
poetry run python -m analysis.rq2.attainment_figures render-figures \
    --dataset <dataset> \
    --pareto-csvs experiments/<dataset>/repair/seed_*/*/mutated/analysis/pareto_per_trial.csv \
    --dataset-level
```

Saved under `evaluation/quantitative/<dataset>/rq2/render_figures/`, as two PNGs: `..._ALL_DPS_all_CART.png` (the accuracy-level slices, at Q1/median/Q3) and `..._ALL_DPS_diff_CART.png` (the attainment-difference slices). Use `--baseline-label`/`--baseline-acc-col`/`--baseline-nodes-col`/`--baseline-jaccard-col` to compare against C4.5 or REPTree instead of CART. See `--help` for the per-decision-point and representative-decision-point rendering modes (`--all-dps`, and the default mode driven by `rq_table_csvs`).

## Qualitative Evaluation

The Pareto explorer can be launched with:

```bash
poetry run python pareto_explorer/app.py
```

To reproduce the exact qualitative evaluation reported in our paper, the required files and detailed instructions are provided in the [qualitative](https://github.com/KDMG/RulesRepair/tree/main/evaluation/qualitative) folder.

## Contact

For questions or further information, please contact:

| Contributor | Contact |
| :--- | :--- |
| Chiara Gobbi | c.gobbi@pm.univpm.it |
