import os
import pandas as pd
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
from rag_app import ask_rag

JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")

# Raise per-attempt timeout for judge calls to reduce TimeoutError on slower runs.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")


# Load Excel file
df = pd.read_excel("data/rag_testcases.xlsx")


test_cases = []

for _, row in df.iterrows():

    contexts = str(row["retrieval_context"]).split("|")
    rag_response = ask_rag(str(row["input"]))

    # RetrievalQA.invoke may return a dict-like payload; normalize to text.
    if isinstance(rag_response, dict):
        actual_output = str(rag_response.get("result", ""))
    else:
        actual_output = str(rag_response)

    test_case = LLMTestCase(
        input=row["input"],
        actual_output=actual_output,
        expected_output=row["expected_output"],
        retrieval_context=contexts
    )

    test_cases.append(test_case)


# Define metrics
metrics = [
    ContextualPrecisionMetric(threshold=0.7, model=JUDGE_MODEL, async_mode=False),
    ContextualRecallMetric(threshold=0.7, model=JUDGE_MODEL, async_mode=False),
    AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL, async_mode=False),
    FaithfulnessMetric(
        threshold=0.7,
        model=JUDGE_MODEL,
        async_mode=False,
        truths_extraction_limit=6,
    ),
]


# Run evaluation
evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(run_async=False, throttle_value=1.0, max_concurrent=1)
)