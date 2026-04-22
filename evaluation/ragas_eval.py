import os
import re
import json
import asyncio
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
)

load_dotenv()

_RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
_META_COLS = [
    "question_id", "Configuration", "Chunker", "Embedder", "LLM",
    "latency", "cost", "source_attribution", "prompt_tokens", "completion_tokens",
]

# ── Mistral response-cleaning helpers ────────────────────────────────────────
# Mistral sometimes wraps JSON in markdown fences (```json...```), which breaks
# RAGAS's internal JSON parser. We proxy the response to strip fences.
# We monkey-patch AsyncOpenAI.chat.completions.create directly so the object
# stays a real AsyncOpenAI instance — required for instructor's isinstance check,
# which RAGAS uses to decide between AsyncInstructor vs Instructor (sync).

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class _CleanMessage:
    def __init__(self, original):
        self._orig = original
        content = original.content or ""
        m = _FENCE_RE.search(content.strip())
        self.content = m.group(1).strip() if m else content

    def __getattr__(self, name):
        return getattr(self._orig, name)


class _CleanChoice:
    def __init__(self, original):
        self._orig = original
        self.message = _CleanMessage(original.message)

    def __getattr__(self, name):
        return getattr(self._orig, name)


class _CleanResponse:
    def __init__(self, original):
        self._orig = original
        self.choices = [_CleanChoice(c) for c in original.choices]

    def __getattr__(self, name):
        return getattr(self._orig, name)


def _patch_client(client: AsyncOpenAI) -> AsyncOpenAI:
    """Monkey-patch chat.completions.create on the instance to strip fences and retry 429s."""
    orig_create = client.chat.completions.create

    async def _clean_create(*args, **kwargs):
        for attempt in range(4):
            try:
                resp = await orig_create(*args, **kwargs)
                return _CleanResponse(resp)
            except Exception as exc:
                msg = str(exc)
                is_rate_limit = "429" in msg or "rate_limit" in type(exc).__name__.lower()
                if is_rate_limit and attempt < 3:
                    wait = 5 * (2 ** attempt)
                    print(f"    Rate limited (429). Retrying in {wait}s…")
                    await asyncio.sleep(wait)
                else:
                    raise

    # Shadow the method on the instance (not the class) so isinstance checks still pass.
    client.chat.completions.create = _clean_create
    return client


# ── Cost helper ───────────────────────────────────────────────────────────────

def calculate_cost(llm_name, prompt_tokens, completion_tokens):
    prices = {
        "Qwen-2.5-14B": {"in": 0.100, "out": 0.100},
        "Llama-3.3-70B": {"in": 0.120, "out": 0.120},
        "Mistral-Large-2": {"in": 2.000, "out": 6.000},
    }
    p = prices.get(llm_name, {"in": 0, "out": 0})
    return (prompt_tokens * p["in"] + completion_tokens * p["out"]) / 1_000_000


# ── Per-sample scorer ─────────────────────────────────────────────────────────

async def score_sample(faithfulness_m, answer_relevancy_m, context_recall_m, context_precision_m, row):
    user_input = str(row.get("question", ""))
    response = str(row.get("answer", "")) if row.get("answer") else ""
    retrieved_contexts = [str(c) for c in (row.get("contexts") or [])]
    reference = str(row.get("ground_truth", "")) if row.get("ground_truth") else ""

    scores = {}

    try:
        result = await faithfulness_m.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        scores["faithfulness"] = result.value
    except Exception as e:
        import traceback
        print(f"    Faithfulness failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        scores["faithfulness"] = np.nan

    try:
        result = await answer_relevancy_m.ascore(user_input=user_input, response=response)
        scores["answer_relevancy"] = result.value
    except Exception as e:
        import traceback
        print(f"    AnswerRelevancy failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        scores["answer_relevancy"] = np.nan

    try:
        result = await context_recall_m.ascore(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        scores["context_recall"] = result.value
    except Exception as e:
        import traceback
        print(f"    ContextRecall failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        scores["context_recall"] = np.nan

    try:
        result = await context_precision_m.ascore(
            user_input=user_input,
            reference=reference,
            retrieved_contexts=retrieved_contexts,
        )
        scores["context_precision"] = result.value
    except Exception as e:
        import traceback
        print(f"    ContextPrecision failed: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        scores["context_precision"] = np.nan

    return scores


# ── Main evaluation loop ──────────────────────────────────────────────────────

async def evaluate_results():
    input_file = "evaluation/results/evaluation_dataset.json"
    per_question_file = "evaluation/results/per_question_scores.json"

    if not os.path.exists(input_file):
        print(f"Data not found at {input_file}. Run benchmark_loop.py first.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    df["cost"] = df.apply(
        lambda row: calculate_cost(row["LLM"], row["prompt_tokens"], row["completion_tokens"]),
        axis=1,
    )
    df["source_attribution"] = df.apply(
        lambda row: 1.0 if str(row.get("expected_source", "")) in str(row.get("answer", "")) else 0.0,
        axis=1,
    )

    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    # Create a real AsyncOpenAI instance (required for instructor's isinstance check),
    # then patch its create method to strip Mistral's markdown fences.
    async_client = _patch_client(
        AsyncOpenAI(base_url="https://api.mistral.ai/v1", api_key=mistral_api_key)
    )

    evaluator_llm = llm_factory("mistral-large-2407", provider="openai", client=async_client)
    evaluator_embeddings = embedding_factory("openai", model="mistral-embed", client=async_client)

    faithfulness_m = Faithfulness(llm=evaluator_llm)
    answer_relevancy_m = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_recall_m = ContextRecall(llm=evaluator_llm)
    context_precision_m = ContextPrecision(llm=evaluator_llm)

    leaderboard = []
    all_per_question = []

    for config in df["Configuration"].unique():
        print(f"\nEvaluating: {config}")
        subset = df[df["Configuration"] == config].reset_index(drop=True)

        rows_scores = []
        for _, row in subset.iterrows():
            print(f"  [{row['question_id']}]")
            scores = await score_sample(
                faithfulness_m, answer_relevancy_m, context_recall_m, context_precision_m, row
            )
            rows_scores.append(scores)

        scores_df = pd.DataFrame(rows_scores)
        for col in _META_COLS:
            if col in subset.columns:
                scores_df[col] = subset[col].values

        all_per_question.append(scores_df)

        # Average each metric over all questions; partial data (some NaN) still produces a result.
        def _mean(col):
            series = scores_df[col]
            return round(series.mean(), 4) if not series.isna().all() else np.nan

        leaderboard.append({
            "Configuration": config,
            "Chunker": subset.iloc[0]["Chunker"],
            "Embedder": subset.iloc[0]["Embedder"],
            "LLM": subset.iloc[0]["LLM"],
            "Latency (s)": round(subset["latency"].mean(), 3),
            "Cost per Query ($)": round(subset["cost"].mean(), 6),
            "Source Attribution": round(subset["source_attribution"].mean(), 2),
            "Faithfulness": _mean("faithfulness"),
            "Answer Relevancy": _mean("answer_relevancy"),
            "Context Recall": _mean("context_recall"),
            "Context Precision": _mean("context_precision"),
        })

    # ── Save per-question scores ──────────────────────────────────────────────
    if all_per_question:
        df_pq = pd.concat(all_per_question, ignore_index=True)
        meta_present = [c for c in _META_COLS if c in df_pq.columns]
        ragas_present = [c for c in _RAGAS_METRICS if c in df_pq.columns]
        other = [c for c in df_pq.columns if c not in meta_present + ragas_present]
        df_pq = df_pq[meta_present + ragas_present + other]

        os.makedirs("evaluation/results", exist_ok=True)
        df_pq.to_json(per_question_file, orient="records", force_ascii=False, indent=2)
        print(f"\nSaved {len(df_pq)} per-question score rows to {per_question_file}")

    # ── Save leaderboard ──────────────────────────────────────────────────────
    df_leaderboard = pd.DataFrame(leaderboard)
    os.makedirs("evaluation/results", exist_ok=True)
    df_leaderboard.to_csv("evaluation/results/metrics_leaderboard.csv", index=False, sep=";")
    print("Leaderboard saved to evaluation/results/metrics_leaderboard.csv")

    print("\n--- Leaderboard Summary ---")
    print(df_leaderboard.to_string())


if __name__ == "__main__":
    asyncio.run(evaluate_results())
