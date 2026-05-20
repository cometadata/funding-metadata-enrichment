import sys
from pathlib import Path

JOB_DIR = Path(__file__).resolve().parents[1]
if str(JOB_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_DIR))
