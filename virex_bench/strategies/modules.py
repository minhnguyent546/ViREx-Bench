from collections.abc import Sequence
from typing import Any

import dspy
from pydantic.fields import FieldInfo

from virex_bench.models.base import BaseLM


class ThinkingCaptureLM(BaseLM):
    def __init__(self, base_lm: BaseLM):
        """
        A wrapper around a base language model that captures the last reasoning content and response.
        """
        object.__setattr__(self, "_base_lm", base_lm)
        object.__setattr__(self, "last_thinking", "")
        object.__setattr__(self, "last_messages", None)
        object.__setattr__(self, "last_response", "")

    def __getattr__(self, name: str):
        return getattr(self._base_lm, name)

    def __setattr__(self, name: str, value):
        if name in ("_base_lm", "last_thinking", "last_messages", "last_response"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._base_lm, name, value)

    def __call__(
        self,
        prompt: str | None = None,
        messages: Sequence[dict[str, str]] | None = None,
        **kwargs,
    ):
        object.__setattr__(self, "last_messages", messages)
        result = self._base_lm(messages=messages, prompt=prompt, **kwargs)
        for output in result:
            if isinstance(output, dict):
                if "text" in output:
                    object.__setattr__(self, "last_response", output["text"])
                elif "content" in output:
                    object.__setattr__(self, "last_response", output["content"])
                if "reasoning_content" in output:
                    object.__setattr__(self, "last_thinking", output["reasoning_content"])
                    break
            elif isinstance(output, str):
                object.__setattr__(self, "last_response", output)
        return result

    async def acall(
        self,
        prompt: str | None = None,
        messages: Sequence[dict[str, str]] | None = None,
        **kwargs,
    ):
        object.__setattr__(self, "last_messages", messages)
        result = await self._base_lm.acall(messages=messages, prompt=prompt, **kwargs)
        for output in result:
            if isinstance(output, dict):
                if "text" in output:
                    object.__setattr__(self, "last_response", output["text"])
                elif "content" in output:
                    object.__setattr__(self, "last_response", output["content"])
                if "reasoning_content" in output:
                    object.__setattr__(self, "last_thinking", output["reasoning_content"])
                    break
            elif isinstance(output, str):
                object.__setattr__(self, "last_response", output)
        return result


class ChainOfThought(dspy.Module):
    def __init__(
        self,
        signature: str | type[dspy.Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ):
        """
        A module that reasons step by step in order to predict the output of a task.

        Args:
            signature (Type[dspy.Signature]): The signature of the module.
            rationale_field (Optional[Union[dspy.OutputField, pydantic.fields.FieldInfo]]): The field that will contain the reasoning.
            rationale_field_type (Type): The type of the rationale field.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = dspy.ensure_signature(signature)  # type: ignore
        desc = "${reasoning}"
        if rationale_field is not None and rationale_field.annotation is not None:
            rationale_field_type = rationale_field.annotation  # type: ignore
        rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)
        extended_signature = signature.prepend(  # type: ignore
            name="reasoning", field=rationale_field, type_=rationale_field_type
        )  # type: ignore
        self.predict = dspy.Predict(extended_signature, **config)  # type: ignore

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)


class DualTask2ChainOfThought(ChainOfThought):
    """
    Dual-stream ChainOfThought that captures both prompted reasoning and native thinking.

    Sets two fields on the result:
    - ``reasoning`` (str) — prompted chain-of-thought, parsed from the LM text output
    - ``thinking_content`` (str) — native model reasoning (e.g. ``<think>`` / ``<thinking>``
      tags) from the model's ``reasoning_content`` API field (enabled via ``enable_thinking``
      or equivalent)

    Important: do NOT set ``rationale_field_type=dspy.Reasoning`` when using this module.
    ``dspy.Reasoning`` consumes the same native reasoning stream, which would make
    ``reasoning`` and ``thinking_content`` identical. Use the default ``str`` to keep
    prompted CoT and native thinking as separate, distinct streams.
    """

    def forward(self, **kwargs):
        lm = kwargs.pop("lm", self.predict.lm) or dspy.settings.lm
        if lm is None:
            return super().forward(**kwargs)

        capture_lm = ThinkingCaptureLM(lm)  # pyright: ignore[reportArgumentType]
        result = super().forward(lm=capture_lm, **kwargs)

        if capture_lm.last_thinking:
            result["thinking_content"] = capture_lm.last_thinking
        if capture_lm.last_messages is not None:
            result["_lm_messages"] = capture_lm.last_messages
        if capture_lm.last_response:
            result["_lm_response"] = capture_lm.last_response
        return result

    async def aforward(self, **kwargs):
        lm = kwargs.pop("lm", self.predict.lm) or dspy.settings.lm
        if lm is None:
            return await super().aforward(**kwargs)

        capture_lm = ThinkingCaptureLM(lm)  # pyright: ignore[reportArgumentType]
        result = await super().aforward(lm=capture_lm, **kwargs)

        if capture_lm.last_thinking:
            result["thinking_content"] = capture_lm.last_thinking
        if capture_lm.last_messages is not None:
            result["_lm_messages"] = capture_lm.last_messages
        if capture_lm.last_response:
            result["_lm_response"] = capture_lm.last_response
        return result
