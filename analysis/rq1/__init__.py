"""RQ1 dominance analysis: how often, and by how much, does RulesRepair's
Pareto front dominate / get dominated by the CART/C4.5/REPTree baselines.

Sub-commands (run via `python -m analysis.rq1 <command>`):
  multi-baseline        coverage of RulesRepair's front by each baseline's single point
  cart-summary          per-trial fraction of the front dominated by CART (D_i) + multi-algorithm table
  dominance-advantage   size of the gap for trials where a baseline dominates RulesRepair
  build-table           combine dominance-advantage's per-dataset outputs into the paper table
"""
