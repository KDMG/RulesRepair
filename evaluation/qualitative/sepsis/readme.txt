The Data Petri Net used for the qualitative evaluation is composed of the Petri net in pn_normative_sepsis.pnml,
the decision point guards stored in normative_model_sepsis_qualitative.pkl, and the observation instances
derived from experiments/sepsis/sepsis_cut/sepsis_train.xes.

The qualitative evaluation files were obtained as follows:
- Data Petri net:
    - Petri net: pn_normative_sepsis.pnml is taken from experiments/sepsis/sepsis_cut/pn_normative.pnml.
    - Guards: starting from experiments/sepsis/decision_points/normative_model.pkl, we constructed the
      guards used in the qualitative evaluation by replacing the normative tree at each decision point
      with a selected repaired tree obtained from the experiments with seed 3:
        - p_4: s3_regrow_leaf_0 (regrow_leaf, 13 nodes), source: experiments/sepsis/repair/seed_3/p_4/mutated/trees.pkl
        - p_8: s3_regrow_internal_0 (regrow_internal, 9 nodes), source: experiments/sepsis/repair/seed_3/p_8/mutated/trees.pkl
        - p_13: s3_regrow_internal_0 (regrow_internal, 9 nodes), source: experiments/sepsis/repair/seed_3/p_13/mutated/trees.pkl
        - p_18: s3_branch_swap_0 (branch_swap, 9 nodes), source: experiments/sepsis/repair/seed_3/p_18/mutated/trees.pkl
        - p_22: s3_change_feature_0 (change_feature, 9 nodes), source: experiments/sepsis/repair/seed_3/p_22/mutated/trees.pkl
        - p_23: s3_regrow_leaf_0 (regrow_leaf, 13 nodes), source: experiments/sepsis/repair/seed_3/p_23/mutated/trees.pkl
        - p_25: original normative tree (no mutation, 1 node), source: experiments/sepsis/decision_points/normative_model.pkl
        - p_27: s3_change_label_0 (change_label, 9 nodes), source: experiments/sepsis/repair/seed_3/p_27/mutated/trees.pkl
        - p_31: s3_regrow_leaf_0 (regrow_leaf, 15 nodes), source: experiments/sepsis/repair/seed_3/p_31/mutated/trees.pkl
        - p_35: s3_branch_swap_0 (branch_swap, 7 nodes), source: experiments/sepsis/repair/seed_3/p_35/mutated/trees.pkl
        - p_39: s3_branch_swap_0 (branch_swap, 5 nodes), source: experiments/sepsis/repair/seed_3/p_39/mutated/trees.pkl
        - p_41: s3_regrow_internal_0 (regrow_internal, 11 nodes), source: experiments/sepsis/repair/seed_3/p_41/mutated/trees.pkl
        - p_42: s3_change_feature_0 (change_feature, 9 nodes), source: experiments/sepsis/repair/seed_3/p_42/mutated/trees.pkl
        The resulting guards are stored in normative_model_sepsis_qualitative.pkl.
- Event log: from sepsis_train.xes is taken from experiments/sepsis/sepsis_cut/sepsis_train.xes.

