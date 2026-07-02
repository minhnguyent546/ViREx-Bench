"""Search-algorithm registry for ToT.

Mirrors the :mod:`virex_bench.strategies.registry` idiom: a name -> class map
plus ``build_search`` / ``list_search`` helpers. New algorithms (DFS, MCTS) are
registered here as they land.
"""

from virex_bench.strategies.tot.search.base import SearchConfig, SearchResult, ThoughtSearch
from virex_bench.strategies.tot.search.beam import BeamSearch
from virex_bench.strategies.tot.search.dfs import DFSSearch
from virex_bench.strategies.tot.search.mcts import MCTSSearch

SEARCH_REGISTRY: dict[str, type[ThoughtSearch]] = {
    BeamSearch.name: BeamSearch,
    DFSSearch.name: DFSSearch,
    MCTSSearch.name: MCTSSearch,
}


def build_search(algorithm: str, config: SearchConfig) -> ThoughtSearch:
    """Instantiate the search algorithm registered under ``algorithm``."""
    normalized_algorithm = algorithm.lower()
    if normalized_algorithm not in SEARCH_REGISTRY:
        available = ", ".join(sorted(SEARCH_REGISTRY))
        raise KeyError(f"Unknown search algorithm {algorithm!r}. Available: {available}")
    return SEARCH_REGISTRY[normalized_algorithm](config)


def list_search() -> list[str]:
    """Return the sorted names of all registered search algorithms."""
    return sorted(SEARCH_REGISTRY)


__all__ = [
    "SEARCH_REGISTRY",
    "SearchConfig",
    "SearchResult",
    "ThoughtSearch",
    "build_search",
    "list_search",
]
