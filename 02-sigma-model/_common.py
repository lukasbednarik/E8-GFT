"""Shared infrastructure for scripts.

Provides a unified test-reporting class, banner printer, and repository
root bootstrapper so that each script does not have to re-implement them.
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_repo_root() -> Path:
    """Return the absolute repository root and ensure ``e8sim`` is importable.

    The repo root is the first ancestor directory of this file that
    contains an ``e8sim/`` subpackage.  This is robust to the script
    living either in the legacy ``debug_plan/scripts/`` location or in
    the per-paper ``__PAPERS__/<paper>/scripts/<paper>/`` layout.
    """
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "e8sim").is_dir():
            root = candidate
            break
    else:
        raise RuntimeError(
            f"Could not locate the e8sim package above {here}. "
            "Expected an ancestor directory containing an `e8sim/` subdirectory."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# Auto-bootstrap on import so that submodules of e8sim are importable from
# this module (single source of truth for shared constants below).
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
