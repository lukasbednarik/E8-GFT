"""Shared helpers for the verification scripts of

  ``Notes on the cosmological constant in E_8 group field theory.''

The scripts in this folder import via ``from _common import ...``;
``bootstrap_repo_root`` puts the directory of this module on ``sys.path``
so that the locally vendored ``e8sim`` package can be imported as a
top-level module.

Provides
--------
- ``bootstrap_repo_root()`` — ensure ``e8sim`` is importable and return
  the directory in which this module lives.
- ``constants_path(root)`` — absolute path to the bundled
  ``e8sim/e8_constants.pt`` file.
- ``Result`` — small PASS/FAIL/SKIP accumulator.
- ``banner(s, w)`` — section banner printer.
- ``DIM_E8`` — re-export from ``e8sim.eix``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_root() -> Path:
    """Return the directory containing this module and ensure ``e8sim``
    is importable from it.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


bootstrap_repo_root()

from e8sim.eix import DIM_E8  # noqa: E402,F401

CONSTANTS_REL = Path("e8sim") / "e8_constants.pt"


def constants_path(root: Path | None = None) -> Path:
    if root is None:
        root = bootstrap_repo_root()
    return root / CONSTANTS_REL


class Result:
    """Accumulate PASS / FAIL / SKIP test outcomes."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.records: list[tuple[str, str, str]] = []

    def report(self, name: str, ok: bool, detail: str = "") -> None:
        marker = "PASS" if ok else "FAIL"
        line = f"  [{marker}] {name}"
        if detail:
            line += f" — {detail}"
        print(line, flush=True)
        self.records.append((marker, name, detail))
        if ok:
            self.passed += 1
        else:
            self.failed += 1

    def skip(self, name: str, reason: str) -> None:
        self.skipped += 1
        print(f"  [SKIP] {name} — {reason}", flush=True)
        self.records.append(("SKIP", name, reason))

    def summary(self, label: str = "") -> str:
        parts = [f"{self.passed} PASS", f"{self.failed} FAIL"]
        if self.skipped:
            parts.append(f"{self.skipped} SKIP")
        s = " / ".join(parts)
        if label:
            s = f"{label}: {s}"
        return s


def banner(title: str, width: int = 70) -> None:
    bar = "=" * width
    print(f"\n{bar}\n{title}\n{bar}", flush=True)
