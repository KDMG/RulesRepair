MIN_SAMPLES_LEAF_PCT_GRID = (0, 1, 2, 5, 10, 20, 50)
CART_CRITERIA = ("gini", "entropy")

DEFAULT_GROW_PARAMS = {"algorithm": "cart", "criterion": "gini", "min_samples_leaf": 0.02}


def to_min_samples_leaf(pct):
    if pct == 0:
        return 1
    return pct / 100


def _config(algorithm, criterion, pct):
    name = f"{algorithm}_{criterion}_msl{pct}" if criterion else f"{algorithm}_msl{pct}"
    return {
        "name": name, "algorithm": algorithm, "criterion": criterion,
        "min_samples_leaf_pct": pct, "min_samples_leaf": to_min_samples_leaf(pct),
    }


def _grow_configs():
    configs = []
    for pct in MIN_SAMPLES_LEAF_PCT_GRID:
        for criterion in CART_CRITERIA:
            configs.append(_config("cart", criterion, pct))
        configs.append(_config("j48", None, pct))
        configs.append(_config("reptree", None, pct))
    return tuple(configs)


GROW_CONFIGS = _grow_configs()


def grow_kwargs(config):
    kwargs = {"algorithm": config["algorithm"], "min_samples_leaf": config["min_samples_leaf"]}
    if config["algorithm"] == "cart":
        kwargs["criterion"] = config["criterion"]
    return kwargs
