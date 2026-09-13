"""RQ2 statistical significance testing: paired Wilcoxon/Hodges-Lehmann/
Cohen's d_z on per-seed IGD+ (RulesRepair vs a baseline).

Sub-commands (run via `python -m analysis.rq2.statistical <command>`):
  igd-comparison         primary per-decision-point test (one baseline)
  rq-table               unified RQ1+RQ2 LaTeX table (one baseline)
  multi-baseline-table   same table against CART/J48/REPTree at once
  holm-table             cross-baseline Holm-Bonferroni correction on top of multi-baseline-table's output
"""
