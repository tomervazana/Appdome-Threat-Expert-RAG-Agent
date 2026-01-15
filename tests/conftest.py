import sys
from pathlib import Path

# Ensure project root is importable (crawler.py, ingest.py, chat.py live there).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
