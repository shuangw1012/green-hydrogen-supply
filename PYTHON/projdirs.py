"""Project directory definitions.

Paths are resolved relative to this file rather than the current working
folder, so the scripts can be launched from a local terminal, an IDE, or an
NCI batch job without changing directory first.
"""

from pathlib import Path
import os

PYTHON_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PYTHON_DIR.parent
DATA_DIR = PROJECT_DIR / "DATA"
MINIZINC_DIR = PROJECT_DIR / "MINIZINC"
OUTPUT_DIR = DATA_DIR / "OPT_OUTPUTS"

# Backward-compatible string paths used by the existing model code.
datadir = str(DATA_DIR) + os.sep
optdir = str(MINIZINC_DIR) + os.sep
