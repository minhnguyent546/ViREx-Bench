"""Small shared helpers used across the package."""

from collections.abc import Sequence


def premises_to_text(premises: Sequence[str]) -> str:
    """Format a list of premises as a numbered block of text.

    Each premise is labeled with its 1-based index so the model can cite
    ``supporting_premise_indices`` unambiguously (premise 1 is the first premise
    in the list). Keeping the premises as a ``list[str]`` in the data layer and
    only collapsing to text at the model-input boundary preserves order while
    making the premise numbering explicit in the prompt.
    """
    return "\n".join(f"Premise {index + 1}. {premise}" for index, premise in enumerate(premises))
