import sys
from pathlib import Path

# Ensure the contentguard root is on sys.path so that
# `import models` and `from server.grader import ...` resolve correctly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
