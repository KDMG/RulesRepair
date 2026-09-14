import tempfile
from pathlib import Path
import re

TMP_DIR = Path(tempfile.mkdtemp(prefix="pareto_explorer_"))

_DP_NAME_RE = re.compile(r"^([A-Za-z]+)_?(\d+)$")


def _dp_display_html(dp_name):
    if dp_name is None:
        return ""
    m = _DP_NAME_RE.match(str(dp_name))
    if not m:
        return str(dp_name)
    prefix, digits = m.groups()
    return f"{prefix}<sub>{digits}</sub>"
