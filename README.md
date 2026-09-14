# RulesRepair

Repairing decision rules in data-aware process models with Keep-Regrow.

## Requirements

To run our application you need to have installed:

* [Python 3.9 to 3.12](https://www.python.org/downloads/)

* [Poetry](https://python-poetry.org/docs/#installation), to install the dependencies listed in `pyproject.toml`

* [Graphviz](https://graphviz.org/download/), required by the `graphviz` Python package to render trees

* (optional, only for the Weka J48/REPTree baselines) a JVM, [OpenJDK 8 or later](https://adoptium.net/)

## Reproduce results

To run our program copy and paste the following command in your terminal:
```
git clone https://github.com/KDMG/RulesRepair/
cd RulesRepair/

poetry env use python3.9   #any Python version >=3.9 and <=3.12
poetry install
source $(poetry env info --path)/bin/activate
poetry run python run_pipeline.py --dataset sepsis --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

## Contact

For any information, please contact:

| Contributor name | Contacts |
| :-------- | :------- |
| `Chiara Gobbi` | c.gobbi@pm.univpm.it |
