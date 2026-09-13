"""Shared small validation-report helper.

Used by the RQ1 dominance/ and RQ2 statistical/ analysis packages to
accumulate PASS/FAIL checks and warnings while building a table, then print
one consolidated report at the end of a run.
"""


class ValidationReport:
    def __init__(self):
        self.checks = []
        self.warnings = []

    def check(self, label, passed, detail=""):
        self.checks.append((label, passed, detail))

    def warn(self, message):
        self.warnings.append(message)

    def print_report(self):
        print("\nvalidation report")
        n_failed = 0
        for label, passed, detail in self.checks:
            status = "pass" if passed else "fail"
            if not passed:
                n_failed += 1
            print(f"  [{status}] {label}{(': ' + detail) if detail else ''}")
        if self.warnings:
            print(f"\n  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                print(f"    [warn] {w}")
        else:
            print("\n  0 warnings.")
        print(f"\noverall: {'all checks passed' if n_failed == 0 else f'{n_failed} check(s) failed, see above'}")
        return n_failed == 0
