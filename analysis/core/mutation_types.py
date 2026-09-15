MUTATION_TYPES = (
    "prune",
    "change_label",
    "branch_swap",
    "change_threshold",
    "change_feature",
    "regrow_leaf",
    "regrow_internal",
)


def get_operator_series(df):
    if "operator_detail" in df.columns:
        return df["operator_detail"]
    if "operator" in df.columns:
        return df["operator"]
    return None


def validate_mutation_types(series, source=""):
    values = set(series.dropna().unique())
    unknown = values - set(MUTATION_TYPES)
    if unknown:
        raise ValueError(f"{source}: found unrecognized operator value(s) {sorted(unknown)}; expected one of {MUTATION_TYPES}.")


def filter_by_mutation_type(df, mutation_type, source=""):
    if mutation_type not in MUTATION_TYPES:
        raise ValueError(f"Unknown mutation_type {mutation_type!r}; expected one of {MUTATION_TYPES}.")
    series = get_operator_series(df)
    if series is None:
        raise ValueError(
            f"error."
        )
    validate_mutation_types(series, source=source)
    return df[series == mutation_type]
