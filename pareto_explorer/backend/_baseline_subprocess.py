import os
import pickle
import sys
from pathlib import Path

PARETO_EXPLORER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = PARETO_EXPLORER_DIR.parent
if str(PARETO_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(PARETO_EXPLORER_DIR))
os.chdir(str(REPO_ROOT))

from backend.live_repair import compute_baseline_point_for_tree


def main():
    input_path, output_path = sys.argv[1], sys.argv[2]
    with open(input_path, "rb") as f:
        prefix, base_tree, df_adapt_raw, df_test_raw, max_depth = pickle.load(f)

    try:
        result = compute_baseline_point_for_tree(
            prefix, base_tree, df_adapt_raw, df_test_raw=df_test_raw, max_depth=max_depth,
        )
        payload = ("ok", result)
    except Exception as exc:
        payload = ("error", str(exc))

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main()
