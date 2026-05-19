#!/usr/bin/env python3
"""Runner for all 01-foundations-of-e8-gft verification scripts.

1. Verifies the Python version (requires ≥ 3.10 — the scripts use PEP 604
   union types like ``list[str] | None`` in signatures).
2. Verifies the availability of required libraries (``numpy``, ``torch``) and
   the importability of the sibling ``e8sim`` package, including the presence
   of the data file ``e8_constants.pt``.
3. Reports the CUDA status — *informational only*. The foundation scripts
   run on CPU; CUDA is **not required**.
4. Runs each verification script in a separate process (``subprocess``)
   in the order given by the README and summarizes PASS / FAIL.

Usage::

    python3 run_all.py           # full run
    python3 run_all.py --quick   # fast smoke test (CI-friendly)

The ``--quick`` flag is propagated to scripts that support it
(currently ``e3_antichain_full_sweep.py``).

Return code: 0 if all scripts PASS, 1 otherwise.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

REQUIRED_PYTHON: tuple[int, int] = (3, 10)
REQUIRED_LIBS: tuple[str, ...] = ("numpy", "torch")

# Order matches the README and the paper (§3 → §4 → §6.4 → §6.5–§6.7 → §6.8).
SCRIPTS_ORDER: tuple[str, ...] = (
    # §3 — Action uniqueness (Appendix A2)
    "e0_algebra_base.py",
    "e1_action_form.py",
    "e1_verify_hypotheses.py",
    "e1_open_points.py",
    "e1_o5_plethysm.py",
    # §4 — BEC condensate
    "e2_bec_phase.py",
    # §6.4 — Algebraic 4D sector
    "e3_orbit_selection.py",
    # §6.5–§6.7 — sub-claims (α), (γ), (δ)
    "k3b_volovik_signature.py",
    "k3b_c1_os_reflection_positivity.py",
    "k3b_os_b_full_interacting.py",
    "e3_antichain_full_sweep.py",
    "do5b_eix_log_determinant.py",
    "r4_eix_v_ind_camporesi_higuchi.py",
    # §6.8 — Topological consistency
    "e5_topology.py",
)

EXCLUDE: frozenset[str] = frozenset({"_common.py", "__init__.py"})

QUICK_ARGS: dict[str, list[str]] = {
    "e1_open_points.py": ["--quick"],
    "e3_antichain_full_sweep.py": ["--quick"],
}

THIS_DIR: Path = Path(__file__).resolve().parent
SCRIPTS_ROOT: Path = THIS_DIR.parent
E8_CONSTANTS: Path = SCRIPTS_ROOT / "e8sim" / "e8_constants.pt"


def _hr(width: int = 72) -> str:
    return "=" * width


def _section(title: str) -> None:
    print("\n" + _hr())
    print(title)
    print(_hr(), flush=True)


def _line(marker: str, text: str) -> None:
    print(f"  [{marker:4s}] {text}", flush=True)


def check_python() -> bool:
    info = sys.version_info
    ok = (info.major, info.minor) >= REQUIRED_PYTHON
    req = f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}"
    _line(
        "OK" if ok else "FAIL",
        f"Python {info.major}.{info.minor}.{info.micro} "
        f"({sys.executable}) — required ≥ {req}",
    )
    return ok


def check_libs() -> bool:
    all_ok = True
    for name in REQUIRED_LIBS:
        if importlib.util.find_spec(name) is None:
            _line("FAIL", f"{name} — not installed (pip install {name})")
            all_ok = False
            continue
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            _line("FAIL", f"{name} — import failed: {exc}")
            all_ok = False
            continue
        _line("OK", f"{name} {getattr(mod, '__version__', '?')}")
    return all_ok


def check_e8sim() -> bool:
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    try:
        e8sim = importlib.import_module("e8sim")
    except Exception as exc:
        _line("FAIL", f"e8sim — import failed: {exc}")
        return False
    _line("OK", f"e8sim — {Path(getattr(e8sim, '__file__', '?')).parent}")

    if not E8_CONSTANTS.exists():
        _line("FAIL", f"e8_constants.pt — missing: {E8_CONSTANTS}")
        return False
    _line("OK", f"e8_constants.pt — {E8_CONSTANTS.stat().st_size / 1024.0:.0f} kB")
    return True


def report_cuda() -> None:
    """Report CUDA status. Not required for the foundation scripts."""
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        _line("SKIP", "CUDA — torch is not importable")
        return
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
        cuda_ver = getattr(torch.version, "cuda", "?")
        _line("INFO", f"CUDA available — {n}× [{names}]; torch.cuda={cuda_ver}")
    else:
        _line("INFO", "CUDA unavailable")
    _line("INFO", "Foundation scripts run on CPU — CUDA is not required.")


def discover_scripts() -> list[Path]:
    """Return runnable scripts in README order; new ones are appended alphabetically."""
    runner_name = Path(__file__).name
    present = {
        p.name for p in THIS_DIR.glob("*.py")
        if p.name not in EXCLUDE and p.name != runner_name
    }
    ordered: list[str] = []
    for name in SCRIPTS_ORDER:
        if name in present:
            ordered.append(name)
            present.discard(name)
    ordered.extend(sorted(present))
    return [THIS_DIR / n for n in ordered]


def run_one(script: Path, extra_args: list[str] | None = None) -> tuple[bool, float]:
    cmd = [sys.executable, str(script)] + (extra_args or [])
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=THIS_DIR)
    return proc.returncode == 0, time.time() - t0


def main() -> int:
    quick = "--quick" in sys.argv[1:]

    _section("Foundations of an E_8 GFT — verification script runner")
    print(f"  working directory : {THIS_DIR}")
    print(f"  scripts root      : {SCRIPTS_ROOT}")
    if quick:
        print(f"  mode              : --quick (CI smoke test)")

    _section("[1/3] Environment check")
    py_ok = check_python()
    libs_ok = check_libs()
    e8_ok = check_e8sim()
    report_cuda()
    if not (py_ok and libs_ok and e8_ok):
        print("\n[FAIL] Environment check failed — aborting.")
        return 1

    _section("[2/3] Script discovery")
    scripts = discover_scripts()
    for s in scripts:
        print(f"  - {s.name}")
    print(f"\n  → {len(scripts)} script(s) to run")
    if not scripts:
        print("\n[WARN] No scripts to run.")
        return 0

    _section(f"[3/3] Running {len(scripts)} scripts")
    results: list[tuple[str, bool, float]] = []
    t_total = time.time()
    for i, script in enumerate(scripts, 1):
        extra = QUICK_ARGS.get(script.name, []) if quick else []
        _section(f"[{i}/{len(scripts)}] {script.name}"
                 + (f"  {extra}" if extra else ""))
        ok, dt = run_one(script, extra)
        marker = "PASS" if ok else "FAIL"
        print(f"\n  → [{marker}] {script.name}  ({dt:.1f}s)", flush=True)
        results.append((script.name, ok, dt))

    elapsed = time.time() - t_total

    _section("Summary")
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    width = max(len(name) for name, _, _ in results)
    for name, ok, dt in results:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name:<{width}}  {dt:7.1f} s")
    print(f"\n  {n_pass} PASS / {n_fail} FAIL  (total {elapsed:.1f}s)")
    print(_hr())

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
