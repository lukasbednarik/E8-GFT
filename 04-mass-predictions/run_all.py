"""Run every verification script for Paper 04 (mass predictions).

Usage::

    python run_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "step1_e7_rep_54modes.py",
    "step2_twisted_dirac_am.py",
    "mass_prediction_pipeline.py",
]


def _print_banner(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


def main() -> int:
    _print_banner("Paper 04: Mass predictions — verification suite")

    results: list[tuple[str, int, float]] = []
    for name in SCRIPTS:
        script = HERE / name
        if not script.exists():
            print(f"  [SKIP] {name} — not found")
            continue
        _print_banner(f"Running {name}")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(HERE),
        )
        elapsed = time.time() - t0
        results.append((name, proc.returncode, elapsed))
        status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        print(f"\n  -> {status} in {elapsed:.1f}s", flush=True)

    _print_banner("Summary")
    n_ok = 0
    for name, rc, elapsed in results:
        status = "OK  " if rc == 0 else "FAIL"
        if rc == 0:
            n_ok += 1
        print(f"  [{status}] {name:<40s}  {elapsed:7.1f}s")
    print(f"\n  {n_ok}/{len(results)} script(s) passed.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
