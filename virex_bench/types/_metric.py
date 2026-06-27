from collections.abc import Callable

import dspy

from virex_bench.types._task import ReasoningExample

ReasoningMetric = Callable[[ReasoningExample, dspy.Prediction], float]
