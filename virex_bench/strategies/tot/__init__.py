"""Tree-of-Thoughts strategy package.

Re-exports :class:`ToTStrategy` so ``from virex_bench.strategies.tot import ToTStrategy``
keeps working after the split from the original single-file ``tot.py``.
"""

from virex_bench.strategies.tot.strategy import ToTStrategy

__all__ = ["ToTStrategy"]
