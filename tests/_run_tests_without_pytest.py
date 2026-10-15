"""
Minimal pytest-compatible test runner.

This project's test suite (tests/test_*.py) is written to run under real
pytest (see requirements.txt: pytest==8.2.2) — nothing here is
pytest-specific beyond the standard built-in `tmp_path` fixture, so
`pytest` will run these files unmodified on any machine with network
access to install dependencies.

This script exists only for environments where pytest cannot be
installed (e.g. an offline sandbox), so the suite can still be verified
end-to-end. It is NOT part of the production pipeline.
"""

import importlib
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

TEST_DIR = Path(__file__).parent
TEST_MODULES = sorted(p.stem for p in TEST_DIR.glob("test_*.py"))


def run():
    sys.path.insert(0, str(TEST_DIR.parent))
    total_passed = 0
    total_failed = 0
    failures = []

    for modname in TEST_MODULES:
        module = importlib.import_module(f"tests.{modname}")
        test_funcs = [
            f for name, f in inspect.getmembers(module, inspect.isfunction)
            if name.startswith("test_") and f.__module__ == module.__name__
        ]
        for f in test_funcs:
            sig = inspect.signature(f)
            with tempfile.TemporaryDirectory() as d:
                try:
                    if "tmp_path" in sig.parameters:
                        f(Path(d))
                    else:
                        f()
                    total_passed += 1
                    print(f"PASS  {modname}::{f.__name__}")
                except Exception:
                    total_failed += 1
                    tb = traceback.format_exc()
                    print(f"FAIL  {modname}::{f.__name__}")
                    failures.append((f"{modname}::{f.__name__}", tb))

    print(f"\n{'='*60}")
    print(f"{total_passed} passed, {total_failed} failed")
    if failures:
        print(f"{'='*60}")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")

    return total_failed == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
