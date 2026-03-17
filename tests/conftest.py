# tests/conftest.py
import sys
from pathlib import Path

# Ensure repository root is importable so `import src...` works in pytest.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
