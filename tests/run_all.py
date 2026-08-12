"""Run every ``test_*`` function in ``tests/`` without needing pytest.

pytest is not a dependency of this project, so this runner discovers the test
modules next to it, calls each ``test_*`` function, and reports a per-test
pass/fail table with the traceback of anything that failed. The test modules are
written to pytest conventions as well, so ``pytest tests/`` also works if it is
installed.

Run: PYTHONPATH=. .venv/bin/python tests/run_all.py
     PYTHONPATH=. .venv/bin/python tests/run_all.py losses   # module or test-name filter
     PYTHONPATH=. .venv/bin/python tests/run_all.py fuzzy    # fuzzy seeding tests only
"""

from __future__ import annotations

import importlib
import inspect
import sys
import time
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def discover() -> list[str]:
    """Test module names, in a stable order."""
    return sorted(p.stem for p in TESTS_DIR.glob("test_*.py"))


def run(filter_text: str | None = None) -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    started = time.perf_counter()

    for module_name in discover():
        try:
            module = importlib.import_module(f"tests.{module_name}")
        except Exception:
            label = module_name
            if filter_text and filter_text not in label:
                continue
            failed.append((module_name, traceback.format_exc()))
            print(f"\n{'=' * 72}\n{module_name}\n{'=' * 72}")
            print(f"  [FAIL] import: {traceback.format_exc(limit=1).strip()}")
            continue

        functions = [
            (name, obj)
            for name, obj in vars(module).items()
            if name.startswith("test_") and inspect.isfunction(obj)
            # Only functions defined in this module, not imported helpers.
            and obj.__module__ == module.__name__
        ]
        # Source order reads better than alphabetical for a progress log.
        functions.sort(key=lambda pair: pair[1].__code__.co_firstlineno)

        selected = []
        for name, function in functions:
            label = f"{module_name}::{name}"
            if filter_text and filter_text not in label:
                continue
            selected.append((name, function, label))
        if not selected:
            continue

        print(f"\n{'=' * 72}\n{module_name}\n{'=' * 72}")
        for name, function, label in selected:
            try:
                function()
                passed.append(label)
            except Exception:
                failed.append((label, traceback.format_exc()))
                print(f"  [FAIL] {name}")

    elapsed = time.perf_counter() - started
    print(f"\n{'=' * 72}")
    if failed:
        print(f"FAILURES ({len(failed)})")
        for label, tb in failed:
            print(f"\n--- {label} ---\n{tb}")
    print(
        f"RESULT: {len(passed)} passed, {len(failed)} failed in {elapsed:.1f}s\n{'=' * 72}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else None))
