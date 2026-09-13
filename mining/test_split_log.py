import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mining.split_log import compute_split_bounds


def _sizes(len_log, normative, train, test):
    end_normative, end_train = compute_split_bounds(len_log, normative, train, test)
    return end_normative, end_train - end_normative, len_log - end_train  # (n_normative, n_train, n_test)


def test_odd_total_with_test_zero_gives_exactly_zero_test_traces():
    n_normative, n_train, n_test = _sizes(31509, 0.5, 0.5, 0.0)
    assert n_test == 0, f"got n_test={n_test}, must be exactly 0 when test_fraction=0.0"
    assert n_normative + n_train == 31509
    assert n_normative == 15754  # unchanged: normative's own floor() is untouched by this fix
    assert n_train == 15755      # train absorbs the 1-trace rounding leftover, not test
    print("test_odd_total_with_test_zero_gives_exactly_zero_test_traces: PASS")


def test_even_total_with_test_zero_was_already_correct_unchanged():
    n_normative, n_train, n_test = _sizes(1050, 0.5, 0.5, 0.0)
    assert n_test == 0
    assert n_normative == 525
    assert n_train == 525
    print("test_even_total_with_test_zero_was_already_correct_unchanged: PASS")


def test_nonzero_test_fraction_behaviour_is_unchanged():
    n_normative, n_train, n_test = _sizes(100, 0.5, 0.3, 0.2)
    assert (n_normative, n_train, n_test) == (50, 30, 20)
    print("test_nonzero_test_fraction_behaviour_is_unchanged: PASS")


def test_nonzero_test_fraction_can_still_have_rounding_leftover_in_test():
    n_normative, n_train, n_test = _sizes(101, 0.5, 0.3, 0.2)
    assert (n_normative, n_train, n_test) == (50, 30, 21)
    print("test_nonzero_test_fraction_can_still_have_rounding_leftover_in_test: PASS")


def test_zero_traces_edge_case():
    n_normative, n_train, n_test = _sizes(0, 0.5, 0.5, 0.0)
    assert (n_normative, n_train, n_test) == (0, 0, 0)
    print("test_zero_traces_edge_case: PASS")


if __name__ == "__main__":
    test_odd_total_with_test_zero_gives_exactly_zero_test_traces()
    test_even_total_with_test_zero_was_already_correct_unchanged()
    test_nonzero_test_fraction_behaviour_is_unchanged()
    test_nonzero_test_fraction_can_still_have_rounding_leftover_in_test()
    test_zero_traces_edge_case()
    print("\nAll split_log tests passed.")
