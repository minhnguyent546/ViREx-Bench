from collections import Counter


class DecodingStrategy:
    """Base class for decoding / answer-aggregation strategies.

    A decoding strategy turns multiple candidate answers (e.g. from sampling the
    model several times) into a single final answer.
    """

    name: str = "base"

    def aggregate(self, candidates: list[str]) -> str:
        raise NotImplementedError


class MajorityVote(DecodingStrategy):
    """Pick the most frequent candidate answer."""

    name = "majority-vote"

    def aggregate(self, candidates: list[str]) -> str:
        if len(candidates) == 0:
            return ""
        counts = Counter(candidate.strip() for candidate in candidates)
        return counts.most_common(1)[0][0]
