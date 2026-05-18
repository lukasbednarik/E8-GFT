"""Run every numerical-verification script in this directory in sequence.

Usage::

    python run_all.py

Preflight checks: Python >= 3.10, numpy, torch.
Scripts are executed in lexicographic order; a non-zero exit code is
recorded but does not abort the whole run.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

MIN_PYTHON = (3, 10)
REQUIRED_LIBS = ("numpy", "torch")


def _print_banner(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"ERROR: Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required, "
            f"found {sys.version.split()[0]}."
        )
    print(f"  Python : {sys.version.split()[0]}  ({sys.executable})")


def check_libs() -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_LIBS:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        versions[name] = getattr(mod, "__version__", "unknown")
        print(f"  {name:7s}: {versions[name]}")
    if missing:
        sys.exit(
            "ERROR: missing required libraries: " + ", ".join(missing)
            + "\n  Install with: pip install " + " ".join(missing)
        )
    return versions


def discover_scripts() -> list[Path]:
    scripts: list[Path] = []
    for p in sorted(HERE.glob("*.py")):
        name = p.name
        if name.startswith("_"):
            continue
        if name in {"run_all.py"}:
            continue
        scripts.append(p)
    return scripts


def run_script(script: Path) -> tuple[int, float]:
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(HERE),
    )
    return proc.returncode, time.time() - t0


def main() -> int:
    _print_banner("Preflight")
    check_python()
    check_libs()
    scripts = discover_scripts()

    if not scripts:
        print("\nNo scripts to run.")
        return 0

    print(f"\nDiscovered {len(scripts)} script(s):")
    for s in scripts:
        print(f"  - {s.name}")

    results: list[tuple[str, int, float]] = []
    for script in scripts:
        _print_banner(f"Running {script.name}")
        rc, elapsed = run_script(script)
        results.append((script.name, rc, elapsed))
        status = "OK" if rc == 0 else f"FAIL (exit {rc})"
        print(f"\n  -> {status} in {elapsed:.1f}s", flush=True)

    _print_banner("Summary")
    width = max(len(n) for n, _, _ in results)
    n_ok = 0
    for name, rc, elapsed in results:
        status = "OK  " if rc == 0 else "FAIL"
        if rc == 0:
            n_ok += 1
        print(f"  [{status}] {name:<{width}}  {elapsed:7.1f}s")
    print(f"\n  {n_ok}/{len(results)} script(s) succeeded.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
