"""CLI dispatch for the statistical package -- run as
`python -m analysis.rq2.statistical <command> [args...]`.
"""
import sys

from analysis.rq2.statistical.igd_comparison import main_igd_comparison
from analysis.rq2.statistical.rq_table import main_rq_table
from analysis.rq2.statistical.multi_baseline_table import main_multi_baseline_table
from analysis.rq2.statistical.holm_table import main_holm_table


COMMANDS = {
    "igd-comparison": main_igd_comparison,
    "rq-table": main_rq_table,
    "multi-baseline-table": main_multi_baseline_table,
    "holm-table": main_holm_table,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python -m analysis.rq2.statistical <command> [arguments...]")
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)
    command = sys.argv[1]
    argv = sys.argv[2:]
    COMMANDS[command](argv)


if __name__ == "__main__":
    main()
