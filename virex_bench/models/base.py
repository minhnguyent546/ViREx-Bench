import dspy


class BaseLM(dspy.LM):
    """Project-wide language-model wrapper.

    Subclasses `dspy.LM`, which already handles talking to any OpenAI-compatible
    endpoint through litellm. Owning this thin wrapper keeps `dspy.LM` from leaking
    across the codebase and gives us a single place to add custom behavior later.
    """

    pass
