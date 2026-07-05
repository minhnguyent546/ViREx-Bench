from collections.abc import Mapping
from typing import Any, Literal

import dspy

from virex_bench.evaluation.metrics import predicted_premise_indices, premises_f1
from virex_bench.tasks.base import ReasoningTask
from virex_bench.types import DatasetConfig, ReasoningExample, ScoreComponents, TaskMetadata


class VietnameseLogicalReasoningSignature(dspy.Signature):
    """Solve a Vietnamese logical-reasoning problem.

    Given a set of premises and a question, reason strictly over the premises to
    produce the correct answer to the question. The premises and question are
    written in Vietnamese.

    Source of truth:
    - Use ONLY the information stated in the given premises. The premises are the
      single source of truth.
    - Do NOT rely on outside knowledge, commonsense reasoning, or assumptions not
      entailed by the premises.
    - Do NOT use closed-world reasoning: a claim is not false merely because it is
      not stated. If the premises entail neither a claim nor its negation, the
      answer is uncertain.

    Answer type:
    - First infer `answer_type` from the question. It must be exactly one of:
      "multiple_choice", "yes_no_uncertain", "numeric", "open_ended".
      - multiple_choice: the question lists explicit answer choices in the text
        (e.g. "A.", "(B)", "C)", or "Đáp án nào sau đây ...").
      - yes_no_uncertain: the question asks whether a specific claim holds
        (e.g. "Có phải ...", "Liệu ...", "Có thể kết luận ... không").
      - numeric: the question asks for a count or numeric value
        (e.g. "Bao nhiêu ...", "Có mấy ...").
      - open_ended: the question asks for a derived fact, entity, name, list, or
        short conclusion that is not one of the above.
    - Then format `answer` according to the inferred `answer_type` as described
      below.

    Answer formatting by type:
    - multiple_choice: return ONLY the option label letter(s) — nothing else.
      For a single answer, return one uppercase letter (e.g. "A"). For multiple
      answers, return the letters separated by ", " in alphabetical order
      (e.g. "B, C"). Do NOT include the option text, any explanation, or ANY
      punctuation beyond the ", " separator. The whole answer must consist of
      labels and ", " alone; surrounding text such as "Đáp án là C", "chọn C",
      "C.", or "C - <option text>" is WRONG even if the chosen label is correct.
    - yes_no_uncertain: return EXACTLY one of the three Vietnamese labels "Có",
      "Không", or "Không chắc chắn" — nothing else. Use "Có" when the premises
      entail the claim, "Không" when the premises entail its negation, and
      "Không chắc chắn" when the premises entail neither. Variants such as
      "Đúng", "Vâng", "Sai", "Không rõ", "Không xác định" are WRONG, and so are
      any other-language equivalents ("Yes", "No", "Uncertain") — even though
      they may be semantically equivalent. Capitalization as shown is preferred.
    - numeric: return only the number whenever possible (e.g. "3"). Include
      accompanying text only when it is genuinely required to make the answer
      meaningful (e.g. a unit such as "3 người").
    - open_ended: return a concise answer in Vietnamese (a short phrase or
      sentence) stating the entity, fact, or conclusion entailed by the premises.
      If no requested fact is entailed, return "Không chắc chắn".

    Premises used (final-answer evidence):
    - `premises_used` means final-answer evidence: the MINIMAL set of premises
      necessary to justify the answer you are returning. It is NOT every premise
      you read, inspected, or mentioned while thinking.
    - Include the FULL positive dependency chain: if a rule fires only because of
      some fact premises, cite BOTH the rule and those fact premises. Do not
      return only rules without the facts that activate them, nor only the final
      segment of a chain.
    - EXCLUDE premises that were only used as background, to analyze or reject
      non-selected options/distractors, or that lead to unrelated consequences --
      even if they are true and you inspected them. If you discuss such a premise,
      describe it in words WITHOUT citing its number.
    - Citation rules by answer type:
      * multiple_choice: cite only the minimal chain supporting the SELECTED
        option. Exclude every premise used only to reject other options.
      * yes_no_uncertain = "Có": cite every rule and fact needed to ENTAIL the
        claim.
      * yes_no_uncertain = "Không": cite every rule and fact needed to entail the
        NEGATION of the claim.
      * yes_no_uncertain = "Không chắc chắn": cite only the premises that
        establish the blocking condition or missing link explaining why the claim
        is not derived. Exclude unrelated chains. Do NOT return empty just because
        the answer is uncertain.
      * numeric: cite the premise(s) stating the quantity, plus any rule needed to
        derive it.
      * open_ended: cite the union of the supporting premises for the returned
        derived fact(s) only.
    - ALL-PREMISES WARNING: if your selected set equals every premise in the list,
      this is almost always wrong. Keep all of them only if each one is genuinely
      necessary for the final derivation; otherwise return the smallest sufficient
      set.
    - `supporting_premise_indices` is the primary output: the 1-based indices of
      that minimal set (premise 1 is the first premise in the list), sorted
      ascending and deduplicated.
    - `relevant_premises` is a fallback: the EXACT original text of those same
      premises, copied verbatim (do not rephrase or shorten -- the downstream
      re-matcher needs a verbatim copy to recover the index). The two fields must
      describe the SAME set, in the same order.
    - Return both empty ONLY when no premise participates in deriving the answer.
    """

    premises: str = dspy.InputField(
        desc="The premises as a numbered text block (Premise 1, Premise 2, ...). The only source of truth."
    )
    question: str = dspy.InputField(desc="The question to answer based on the premises.")

    answer_type: Literal["multiple_choice", "yes_no_uncertain", "numeric", "open_ended"] = (
        dspy.OutputField(
            desc=(
                "The type of answer inferred from the question. Exactly one of: "
                "multiple_choice, yes_no_uncertain, numeric, open_ended."
            )
        )
    )
    answer: str = dspy.OutputField(
        desc=(
            "The final answer, formatted according to `answer_type`:\n"
            "- multiple_choice: ONLY option label letter(s), e.g. 'A' or 'B, C' "
            "(uppercase, alphabetical, comma+space separated). No option text, no "
            "explanation, no punctuation beyond ', ' — 'Đáp án là C' / 'C.' are WRONG.\n"
            "- yes_no_uncertain: EXACTLY one of 'Có', 'Không', 'Không chắc chắn' — "
            "no variants ('Đúng', 'Sai', 'Không rõ') and no other languages ('Yes').\n"
            "- numeric: the number only when possible, e.g. '3'; add text only "
            "when truly needed, e.g. '3 người'.\n"
            "- open_ended: a concise Vietnamese phrase or sentence; "
            "'Không chắc chắn' if nothing is entailed."
        )
    )
    supporting_premise_indices: list[int] = dspy.OutputField(
        desc=(
            "PRIMARY source of premises_used: the 1-based indices of the MINIMAL set of "
            "premises that justify `answer` (premise 1 is the first premise). Follow the "
            "per-answer-type and full-dependency-chain rules in the signature. Exclude "
            "premises used only as background or to reject other options. Sort ascending, "
            "deduplicate, e.g. [1, 3, 4]. Empty only if no premise is used. "
            "ALL-PREMISES WARNING: returning every premise is almost always wrong. "
            "EXAMPLES (style only): a Yes answer applies a rule that needs a fact premise "
            "to fire -> cite BOTH, e.g. [4, 9], not [9] alone. An MCQ answer supported by "
            "premises 1, 2, 4 where premise 5 only rejected a distractor -> [1, 2, 4], not "
            "[1, 2, 4, 5]."
        )
    )
    relevant_premises: list[str] = dspy.OutputField(
        desc=(
            "FALLBACK for premises_used: the EXACT original text of the SAME premises listed "
            "in `supporting_premise_indices`, copied verbatim from the premise list (do not "
            "rephrase, shorten, or add numbering). The downstream re-matcher needs a verbatim "
            "copy to recover the correct index. Same set, same order as the indices."
        )
    )


class VietnameseLogicalReasoningAggregationSignature(dspy.Signature):
    """Aggregate multiple candidate answers for a Vietnamese logical-reasoning problem.

    You are an aggregation oracle. You receive N candidate answers produced by
    independent reasoning paths for the same premises and question. Each candidate
    includes the inferred answer type, the answer, the reasoning, the supporting
    premise indices, and the relevant premise texts.

    Your job:
    1. Determine the majority answer using the voting rules below.
    2. Pick the answer_type that is most consistent with the majority answer.
    3. Select the supporting_premise_indices that appear most often among the
       majority candidates.
    4. Synthesize a single explanation that directly justifies the majority answer.

    ## VOTING RULES:

    ### Closed answer types (option labels, yes/no/uncertain, numeric):
    - Match answers exactly after normalization (case-insensitive, whitespace-collapsed).
    - The majority is the value that appears most often.

    ### Open-ended answers (free-form text):
    - Use **semantic equivalence** to group answers, not exact text matching.
    - After grouping, pick the group with the most members.

    ## OUTPUT RULES:
    - ``answer`` MUST exactly match one of the candidate answer values (use the
      original text from the majority candidate, not a paraphrase).
    - ``answer_type`` must match the type inferred by the majority of candidates.
    - ``supporting_premise_indices`` should be the union (deduplicated, sorted) of
      premise indices from the majority candidates, or the most common set if there
      is a clear majority.
    - ``relevant_premises`` must be the EXACT original text of those same premises,
      copied verbatim from the premise list (do not rephrase).
    - ``explanation`` must be synthesized from the majority candidates' reasoning.
      Write it as step-by-step reasoning over the premises that leads to the answer —
      the same style and depth as the individual candidates' ``reasoning`` fields, not
      a concise summary. Cite only what the premises state and avoid outside knowledge.
      Write in the SAME language as the candidate reasoning (Vietnamese). Do not
      mention voting or aggregation.
    """

    premises: str = dspy.InputField(
        desc="The original premises as a numbered text block (Premise 1, Premise 2, ...). The only source of truth."
    )
    question: str = dspy.InputField(desc="The original question.")
    candidate_answers: str = dspy.InputField(
        desc=(
            "JSON array of candidate results. Each entry is an object containing: "
            '"answer" (str), "reasoning" (str), "answer_type" (str), '
            '"supporting_premise_indices" (list[int]), "relevant_premises" (list[str]). '
            'Example: [{"answer": "Có", "reasoning": "...", "answer_type": "yes_no_uncertain", '
            '"supporting_premise_indices": [1, 3], "relevant_premises": ["..."]}, ...].'
        ),
    )
    answer: str = dspy.OutputField(
        desc="The majority-voted final answer. Must exactly match one of the candidate values."
    )
    answer_type: Literal["multiple_choice", "yes_no_uncertain", "numeric", "open_ended"] = (
        dspy.OutputField(
            desc=(
                "The answer type most consistent with the majority answer. Exactly one of: "
                "multiple_choice, yes_no_uncertain, numeric, open_ended."
            ),
        )
    )
    supporting_premise_indices: list[int] = dspy.OutputField(
        desc=(
            "The 1-based indices of the minimal set of premises that justify the majority "
            "answer. Union of the majority candidates' indices, sorted ascending, deduplicated."
        ),
    )
    relevant_premises: list[str] = dspy.OutputField(
        desc=(
            "The EXACT original text of the same premises listed in "
            "supporting_premise_indices, copied verbatim from the premise list."
        ),
    )
    explanation: str = dspy.OutputField(
        desc=(
            "Step-by-step reasoning over the premises that leads to the answer, "
            "synthesized from the majority candidates' reasoning. Same style and depth "
            "as the candidates' reasoning fields — not a concise summary. Cite only "
            "what the premises state, avoid outside knowledge, write in Vietnamese. "
            "Do not mention voting or aggregation."
        ),
    )


class VietnameseLogicalReasoning(ReasoningTask):
    metadata = TaskMetadata(
        name="vietnamese-logical-reasoning",
        description=(
            "Vietnamese logical-reasoning task: given a set of premises and a "
            "question, derive the answer that is logically supported by the "
            "premises. Covers multiple-choice, yes/no/uncertain, numeric, and "
            "short free-text answers."
        ),
        language="vie",
        dataset=DatasetConfig(
            path="minhnguyent546/virex-bench-datasets",
            name="logical-reasoning",
            split="test",
            # revision="712522ef946b72b6d1d7a34d5fbab98696feac54",
            # revision="b7b189fcf8b40c4e12863fce850ba42d37598ca8",
            # revision="d7842200648b2de86711017251c6fb206bf289d5",
            revision="9650d6f970dae5fc73ecbaf3885bd338e447d45d",
            num_proc=2,
        ),
        main_metric="llm_judge",
        judge="logical_reasoning",
    )
    signatures = {
        "default": VietnameseLogicalReasoningSignature,
        "direct": VietnameseLogicalReasoningSignature,
        "cot": VietnameseLogicalReasoningSignature,
        "tot": VietnameseLogicalReasoningSignature,
        # 'pot_z3': TODO,
    }
    rationale_fields = {
        "default": dspy.OutputField(
            desc=(
                "Step-by-step reasoning over the premises that leads to the answer. "
                "Cite only what the premises state and avoid outside knowledge. "
                "IMPORTANT: write this reasoning in Vietnamese."
            )
        ),
        "cr": dspy.OutputField(
            desc=(
                "Before answering, reason over the verified propositions in "
                "`accumulated_context`. First check whether the "
                "[Mệnh đề được xác nhận] (entailed) bucket directly settles the "
                "question — if it does, commit that answer. If the question turns "
                "on a point that only the [Mệnh đề không xác định] (undetermined) "
                "bucket touches, the evidence is INSUFFICIENT — answer "
                "'Không chắc chắn'. Never confuse 'undetermined' with 'is false'. "
                "Write this reasoning in Vietnamese."
            )
        ),
    }
    aggregation_signature = VietnameseLogicalReasoningAggregationSignature

    def _row_to_example(self, row: Mapping[str, Any]) -> ReasoningExample:
        return ReasoningExample(
            example_id=str(row["query_id"]),
            premises=[str(premise) for premise in row["premises"]],
            question=str(row["query"]),
            answer=str(row["answer"]),
            premises_used=[int(index) for index in row.get("premises_used", [])],
        )

    def recorded_inputs(
        self, example: ReasoningExample, model_inputs: dict[str, object]
    ) -> dict[str, object]:
        # Record the gold premises alongside the model inputs so the results file
        # carries the premise-level supervision without leaking it to the model.
        # ``model_inputs["premises"]`` is the numbered text block actually fed to
        # the model (see ReasoningTask.example_to_inputs); restore the original
        # list here so the saved results keep the structured premise representation.
        return {
            **model_inputs,
            "premises": example.premises,
            "premises_used": example.premises_used,
        }

    def compute_score(
        self,
        example: ReasoningExample,
        prediction: dspy.Prediction,
        answer_score: float,
    ) -> ScoreComponents:
        # Blend answer correctness with premise-selection F1 only when the example
        # carries gold premise supervision: final = 0.5 * answer + 0.5 * premises_f1.
        if len(example.premises_used) == 0:
            return ScoreComponents(score=answer_score)
        premises_f1_score = premises_f1(example, prediction)
        return ScoreComponents(
            score=0.5 * answer_score + 0.5 * premises_f1_score,
            llm_judge_score=answer_score,
            premises_f1=premises_f1_score,
            predicted_premises_used=sorted(predicted_premise_indices(example, prediction)),
        )
