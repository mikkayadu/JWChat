import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# A Celery pool process imports the worker inside cwd_in_path(), which drops
# the project root from sys.path afterwards; tasks then import `languages`.
_POOL_PROCESS = f"""
import sys
sys.path[:] = [p for p in sys.path if p not in ("", {str(_ROOT)!r})]
from celery.utils.imports import cwd_in_path
with cwd_in_path():
    import ingestion.tasks.worker
import languages
"""


def test_pool_process_can_import_top_level_modules():
    result = subprocess.run(
        [sys.executable, "-c", _POOL_PROCESS], cwd=_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
