import uuid


def generate_scenario0(df_initial, target):
    return [{
        "trial_id": str(uuid.uuid4()),
        "scenario": 0,
        "feature_type": "none",
        "intensity": 0.0,
        "extent": 0.0,
        "features": [],
        "target_values": {},
        "seed": None,
        "pair_seed": None,
        "perturbation_seed": None,
        "total_possible_pairs": None,
        "pair_selection": None,
    }]
