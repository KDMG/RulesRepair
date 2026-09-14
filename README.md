# RulesRepair

Repairing Decision Rules in Data-Aware Process Models 

## Requirements

To run our application you need to have installed:

* [Python 3.9 to 3.12](https://www.python.org/downloads/)

* [Poetry](https://python-poetry.org/docs/#installation), to install the dependencies listed in `pyproject.toml`

* [Graphviz](https://graphviz.org/download/), required by the `graphviz` Python package to render trees

* [OpenJDK 8 or later](https://adoptium.net/) optional, only for the Weka C4.5/REPTree

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/KDMG/RulesRepair/
cd RulesRepair/

poetry env use python3.9   # any Python version >=3.9 and <=3.12
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

### Qualitative evaluation
You can inspect the Pareto explorer by running 
```bash
python pareto_explorer/app.py
```
you can find the files used to show the qualitative evaluation in the folder 

## Contact

For any information, please contact:

| Contributor name | Contacts |
| :-------- | :------- |
| `Chiara Gobbi` | c.gobbi@pm.univpm.it |
