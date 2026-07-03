"""Cumulative Reasoning strategy package.

Re-exports :class:`CRStrategy` so ``from virex_bench.strategies.cr import CRStrategy``
mirrors the ToT package's public surface.
"""

from virex_bench.strategies.cr.strategy import CRStrategy

__all__ = ["CRStrategy"]
