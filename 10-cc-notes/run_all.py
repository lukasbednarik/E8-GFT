"""Run all numerical verification scripts of

    ``Notes on the cosmological constant in E_8 group field theory.''

Performs a minimal environment check (Python version, required
libraries) and then runs each verification script in sequence with
its default arguments. No command-line parameters; just::

    python3 scripts/run_all.py

The exit code is zero iff every script exits zero.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time
from pathlib import Path


MIN_PYTHON = (3, 9)

REQUIRED_LIBS = (
    ("numpy", None),
    ("torch", None),
)

SCRIPTS = (
    "do5b_camporesi_higuchi_spectral_zeta.py",
    "cc2_f1_camporesi_higuchi.py",
    "cc1_f2_skyrme_bogomolny.py",
)


def banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required; "
            f"have {sys.version.split()[0]}"
        )
    print(f"Python {sys.version.split()[0]} OK")


def check_libs() -> None:
    missing: list[str] = []
    for name, _ in REQUIRED_LIBS:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            missing.append(name)
            continue
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:<8s} {ver}")
    if missing:
        sys.exit(
            "Missing required packages: "
            + ", ".join(missing)
            + ".  Install with:  pip install " + " ".join(missing)
        )


def run_script(script_dir: Path, name: str) -> tuple[str, bool, float]:
    path = script_dir / name
    banner(f"RUN  {name}")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(script_dir),
    )
    dt = time.perf_counter() - t0
    ok = proc.returncode == 0
    status = "PASS" if ok else f"FAIL (exit {proc.returncode})"
    print(f"\n[{status}] {name}  ({dt:.1f}s)")
    return name, ok, dt


def main() -> int:
    banner("Environment")
    check_python()
    check_libs()

    script_dir = Path(__file__).resolve().parent

    results = [run_script(script_dir, name) for name in SCRIPTS]

    banner("Summary")
    width = max(len(n) for n, _, _ in results)
    for name, ok, dt in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {name:<{width}s}  {dt:6.1f}s")

    failed = [n for n, ok, _ in results if not ok]
    if failed:
        print(f"\n{len(failed)} script(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll scripts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
