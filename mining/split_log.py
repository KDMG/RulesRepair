import argparse
import math
from copy import deepcopy
from pathlib import Path
from collections import Counter

import pandas as pd
import pm4py

from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.util.sorting import sort_timestamp_log


def is_missing(value):
    if value is None:
        return True

    try:
        result = pd.isna(value)
        return bool(result) if isinstance(result, bool) else False
    except (TypeError, ValueError):
        return False


def clean_attributes(attributes):
    keys_to_remove = [
        key
        for key, value in attributes.items()
        if is_missing(value)
    ]

    for key in keys_to_remove:
        del attributes[key]


def clean_log(log):
    cleaned_log = deepcopy(log)

    clean_attributes(cleaned_log.attributes)

    for trace in cleaned_log:
        clean_attributes(trace.attributes)

        for event in trace:
            clean_attributes(event)

    return cleaned_log


def compute_split_bounds(len_log, normative_fraction, train_fraction, test_fraction):
    end_normative = math.floor(len_log * normative_fraction)
    end_train = end_normative + math.floor(len_log * train_fraction)
    if test_fraction == 0:
        end_train = len_log
    return end_normative, end_train


def create_log(source_log, traces):
    return EventLog(
        list(traces),
        attributes=deepcopy(source_log.attributes),
        extensions=deepcopy(source_log.extensions),
        classifiers=deepcopy(source_log.classifiers),
        omni_present=deepcopy(source_log.omni_present),
        properties=deepcopy(source_log.properties),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Choose how to split log normative-train-test"
    )

    parser.add_argument(
        "--normative",
        type=float,
        default=0.5,
        help="Normative fraction"
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.5,
        help="Train fraction"
    )

    parser.add_argument(
        "--test",
        type=float,
        default=0,
        help="Test fraction"
    )

    parser.add_argument(
        "--xes",
        default="datasets/sepsis/sepsis.xes",
        help="Event log"
    )

    parser.add_argument(
        "--out-dir", "--out_dir",
        dest="out_dir",
        help="Output directory",
        default="datasets/sepsis_cut",
    )

    parser.add_argument(
        "--name",
        default=None,
        help="Dataset name used to prefix the output files (<name>_normative.xes/"
             "_train.xes/_test.xes). Defaults to the --xes file's own stem (e.g. "
             "'sepsis.xes' -> 'sepsis'), so output files are never ambiguous "
             "between datasets sharing the same --out-dir.",
    )

    args = parser.parse_args()

    log = args.xes
    log = pm4py.read_xes(log)

    dataset_name = args.name if args.name else Path(args.xes).stem

    # Check that the fractions add up
    total_fraction = args.normative + args.train + args.test

    if not math.isclose(total_fraction, 1.0):
        raise ValueError(
            f"The fractions must add up to 1.0, "
            f"but they add up to {total_fraction}"
        )

    sorted_log = sort_timestamp_log(log)

    len_log = len(sorted_log)

    end_normative, end_train = compute_split_bounds(len_log, args.normative, args.train, args.test)

    log_normative = create_log(
        sorted_log,
        sorted_log[:end_normative]
    )

    log_train = create_log(
        sorted_log,
        sorted_log[end_normative:end_train]
    )

    log_test = create_log(
        sorted_log,
        sorted_log[end_train:]
    )

    log_normative = clean_log(log_normative)
    log_train = clean_log(log_train)
    log_test = clean_log(log_test)

    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    pm4py.write_xes(
        log_normative,
        str(output_directory / f"{dataset_name}_normative.xes")
    )

    pm4py.write_xes(
        log_train,
        str(output_directory / f"{dataset_name}_train.xes")
    )

    pm4py.write_xes(
        log_test,
        str(output_directory / f"{dataset_name}_test.xes")
    )

    print(f"Log originale: {len(sorted_log)}")
    print(f"Normative: {len(log_normative)} -> {output_directory / f'{dataset_name}_normative.xes'}")
    print(f"Train: {len(log_train)} -> {output_directory / f'{dataset_name}_train.xes'}")
    print(f"Test: {len(log_test)} -> {output_directory / f'{dataset_name}_test.xes'}")


if __name__ == "__main__":
    main()
