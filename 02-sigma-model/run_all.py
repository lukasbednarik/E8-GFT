"""Run every numerical-verification script in this directory in sequence.

Usage::

    python run_all.py

The runner performs three preflight checks before executing anything:

1. Python interpreter version (>= 3.10 required for ``from __future__
   import annotations`` plus PEP 604 union types used across the
   scripts).
2. Required third-party libraries (``numpy`` and ``torch``).
3. CUDA availability — *only* checked if at least one of the scripts
   actually needs it (currently ``e10_b1_skyrmion_profile.py`` uses
   ``torch`` for the radial relaxation).  CUDA is **not** required;
   the scripts fall back to CPU automatically.  The check is purely
   informational so the user knows which device the GPU-capable
   scripts will use.

The runner then executes each ``e*.py`` script (excluding ``_common``
and itself) in lexicographic order using the *same* Python interpreter
that launched ``run_all.py``.  Each script's stdout/stderr is streamed
live; a non-zero exit code is recorded but does not abort the whole
run, so the final summary lists every script's status.
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
CUDA_USING_SCRIPTS = ("e10_b1_skyrmion_profile.py",)


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


def check_cuda_if_needed(scripts: list[Path]) -> None:
    needs_cuda_capable = any(s.name in CUDA_USING_SCRIPTS for s in scripts)
    if not needs_cuda_capable:
        print("  CUDA   : not required by any script")
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if n else "?"
        print(f"  CUDA   : available ({n} device(s), {name})")
    else:
        print("  CUDA   : not available — GPU-capable scripts will run on CPU")


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
    check_cuda_if_needed(scripts)

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
