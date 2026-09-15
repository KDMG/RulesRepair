"""CLI dispatch for the dominance package -- run as
`python -m analysis.rq1 <command> [args...]`.
"""
import sys

from analysis.rq1.multi_baseline import main_multi_baseline
from analysis.rq1.dominance_summary import main_dominance_summary
from analysis.rq1.dominance_advantage import main_dominance_advantage


COMMANDS = {
    "multi-baseline": main_multi_baseline,
    "dominance-summary": main_dominance_summary,
    "dominance-advantage": main_dominance_advantage,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python -m analysis.rq1 <command> [arguments...]")
        print(f"Available commands: {', '.join(COMMANDS)}")
        print(f"(--help on each command for its arguments, e.g. "
              f"'python -m analysis.rq1 multi-baseline --help')")
        sys.exit(1)
    command = sys.argv[1]
    argv = sys.argv[2:]
    COMMANDS[command](argv)


if __name__ == "__main__":
    main()
