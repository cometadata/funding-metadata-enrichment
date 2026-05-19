"""Make sibling scripts importable so `from <script_name> import ...` works."""
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parents[1]
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
