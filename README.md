# RulesRepair

Repairing decision rules in data-aware process models with Keep-Regrow.

## Requirements

To run our application you need to have installed:

* [Python 3.8+](https://www.python.org/downloads/)

* [Poetry](https://python-poetry.org/docs/#installation), to install the dependencies listed in `pyproject.toml`

* [Graphviz](https://graphviz.org/download/), required by the `graphviz` Python package to render trees

<<<<<<< Updated upstream
* (optional, only for the Weka C4.5/REPTree baselines) a JVM, OpenJDK 8 or later
=======
* (optional, only for the Weka J48/REPTree baselines) a JVM, [OpenJDK 8 or later](https://adoptium.net/)
>>>>>>> Stashed changes

## Reproduce results

To run our program copy and paste the following command in your terminal:
```
git clone https://github.com/KDMG/RulesRepair/
cd RulesRepair/

poetry install
poetry run python run_pipeline.py --dataset sepsis --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
```

<<<<<<< Updated upstream
=======
`--seeds` only accepts an explicit comma-separated list (no range syntax); the paper's results use 30 seeds, 0 to 29.

The [`datasets`](https://github.com/KDMG/RulesRepair/tree/main/datasets) folder contains the raw event logs; `run_pipeline.py --dataset <name>` runs mining, decision-point extraction, and repair end to end for one of them. Known dataset names: `sepsis`, `production`, `hospital_billing`, `road_traffic`, `prepaid_travel_costs`, `international_declarations`.

>>>>>>> Stashed changes
## Contact

For any information, please contact:

| Contributor name | Contacts |
| :-------- | :------- |
| `Chiara Gobbi` | c.gobbi@pm.univpm.it |
