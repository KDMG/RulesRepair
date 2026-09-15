"""RQ1 dominance analysis: how often, and by how much, does RulesRepair's
Pareto front dominate / get dominated by the CART/C4.5/REPTree baselines.

Sub-commands (run via `python -m analysis.rq1 <command>`):
  multi-baseline        coverage of RulesRepair's front by each baseline's single point
  dominance-summary     per-baseline (CART/C4.5/REPTree) % of trials dominating at least one front point
                         (add -combine to build the paper table across datasets)
  dominance-advantage   size of the accuracy/simplicity/similarity gap for trials where a baseline
                         dominates RulesRepair (add -combine to build the paper table across datasets)
"""
