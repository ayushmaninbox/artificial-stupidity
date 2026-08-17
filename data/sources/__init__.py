"""Corpus sources for Artificial Stupidity."""

import time
from pathlib import Path


class Sink:
    """A budgeted line writer.

    Every source gets a byte budget and writes through one of these, so the
    final corpus mix is something we chose rather than something that happened
    to us based on which scraper was fastest.
    """

    def __init__(self, path: Path, budget_bytes: int, label: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget = budget_bytes
        self.label = label or self.path.stem
        self.written = 0
        self.lines = 0
        self._fh = None
        self._t0 = time.time()

    def __enter__(self):
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._fh:
            self._fh.close()
        return False

    @property
    def full(self) -> bool:
        return self.written >= self.budget

    def write(self, line: str) -> bool:
        """Returns False once the budget is spent."""
        if self.full:
            return False
        data = line + "\n"
        self._fh.write(data)
        self.written += len(data.encode("utf-8"))
        self.lines += 1
        return True

    def progress(self) -> str:
        pct = 100 * self.written / max(1, self.budget)
        return (f"    {self.label:<22} {self.written / 1e6:6.2f} MB "
                f"({pct:5.1f}%)  {self.lines:>8,} lines  "
                f"{time.time() - self._t0:5.0f}s")
