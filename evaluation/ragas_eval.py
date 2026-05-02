import os
import json
import asyncio
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ragas.llms import llm_factory
from scripts.logger import setup_logging
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
  Faithfulness,
  ContextRecall,
  ContextPrecision,
  AnswerRelevancy,
)


load_dotenv()

_RAGAS_METRICS = ["faithfulness", "context_recall", "context_precision", "answer_relevancy"]
_META_COLS = ["question_id", "Configuration", "Chunker", "Embedder", "LLM", "latency", "source_attribution", "prompt_tokens", "completion_tokens"]

_HF_BASE_URL = "https://router.huggingface.co/featherless-ai/v1"
_EVAL_MODEL = "Qwen/Qwen2.5-14B-Instruct"
_CONCURRENCY = 2  # each sample fires 4 concurrent metrics; 2×4=8 stays under Featherless AI's 10-unit limit


# ── Per-sample scorer ─────────────────────────────────────────────────────────

async def score_sample(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, row):
  user_input = str(row.get("question", ""))
  response = str(row.get("answer", "")) if row.get("answer") else ""
  retrieved_contexts = [str(c) for c in (row.get("contexts") or [])]
  reference = str(row.get("ground_truth", "")) if row.get("ground_truth") else ""

  async def _faithfulness():
    try:
      result = await faithfulness_m.ascore(
        user_input=user_input, response=response, retrieved_contexts=retrieved_contexts,
      )
      return result.value
    except Exception as e:
      print(f"Faithfulness failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  async def _context_recall():
    try:
      result = await context_recall_m.ascore(
        user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference,
      )
      return result.value
    except Exception as e:
      print(f"ContextRecall failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  async def _context_precision():
    try:
      result = await context_precision_m.ascore(
        user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts,
      )
      return result.value
    except Exception as e:
      print(f"ContextPrecision failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  async def _answer_relevancy():
    try:
      result = await answer_relevancy_m.ascore(
        user_input=user_input, response=response,
      )
      return result.value
    except Exception as e:
      print(f"AnswerRelevancy failed: {type(e).__name__}: {e}")
      print(traceback.format_exc())
      return np.nan

  faith, recall, precision, relevancy = await asyncio.gather(
    _faithfulness(), _context_recall(), _context_precision(), _answer_relevancy()
  )

  return {
    "faithfulness": faith,
    "context_recall": recall,
    "context_precision": precision,
    "answer_relevancy": relevancy,
  }


# ── Main evaluation loop ──────────────────────────────────────────────────────

async def evaluate_results():
  input_file = "evaluation/results/evaluation_dataset.json"
  per_question_file = "evaluation/results/per_question_scores.json"
  leaderboard_file = "evaluation/results/metrics_leaderboard.csv"

  if not os.path.exists(input_file):
    print(f"Data not found at {input_file}. Run benchmark_loop.py first.")
    return

  with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

  df = pd.DataFrame(data)

  df["source_attribution"] = df.apply(
    lambda row: 1.0 if str(row.get("expected_source", "")) in str(row.get("answer", "")) else 0.0,
    axis=1,
  )

  # Load existing results so we can resume a previously interrupted run.
  os.makedirs("evaluation/results", exist_ok=True)

  existing_pq: list[dict] = []
  if os.path.exists(per_question_file):
    with open(per_question_file, "r", encoding="utf-8") as f:
      existing_pq = json.load(f)

  existing_leaderboard: list[dict] = []
  if os.path.exists(leaderboard_file):
    existing_leaderboard = pd.read_csv(leaderboard_file, sep=";").to_dict("records")

  already_done = {row["Configuration"] for row in existing_leaderboard}

  llm_client = AsyncOpenAI(base_url=_HF_BASE_URL, api_key=os.environ["HF_TOKEN"])
  evaluator_llm = llm_factory(_EVAL_MODEL, provider="openai", client=llm_client)
  evaluator_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

  faithfulness_m = Faithfulness(llm=evaluator_llm)
  context_recall_m = ContextRecall(llm=evaluator_llm)
  context_precision_m = ContextPrecision(llm=evaluator_llm)
  answer_relevancy_m = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

  for config in df["Configuration"].unique():
    if config in already_done:
      print(f"Skipping {config} (already evaluated)")
      continue

    print(f"\nEvaluating: {config}")
    subset = df[df["Configuration"] == config].reset_index(drop=True)

    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _score_row(row):
      async with sem:
        t0 = asyncio.get_event_loop().time()
        scores = await score_sample(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, row)
        elapsed = asyncio.get_event_loop().time() - t0
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] | [{row['question_id']}] evaluated ({elapsed:.1f}s)")
        return scores

    rows_scores = await asyncio.gather(*[_score_row(row) for _, row in subset.iterrows()])

    scores_df = pd.DataFrame(rows_scores)
    for col in _META_COLS:
      if col in subset.columns:
        scores_df[col] = subset[col].values

    meta_present = [c for c in _META_COLS if c in scores_df.columns]
    ragas_present = [c for c in _RAGAS_METRICS if c in scores_df.columns]
    other = [c for c in scores_df.columns if c not in meta_present + ragas_present]
    scores_df = scores_df[meta_present + ragas_present + other]

    # ── Write per-question scores after each config ───────────────────────
    existing_pq.extend(scores_df.to_dict("records"))
    with open(per_question_file, "w", encoding="utf-8") as f:
      json.dump(existing_pq, f, ensure_ascii=False, indent=2)
    print(f"  Saved per-question scores ({len(existing_pq)} rows total)")

    # Average each metric over all questions; partial data (some NaN) still produces a result.
    def _mean(col):
      series = scores_df[col]
      return round(series.mean(), 4) if not series.isna().all() else np.nan

    existing_leaderboard.append({
      "Configuration": config,
      "Chunker": subset.iloc[0]["Chunker"],
      "Embedder": subset.iloc[0]["Embedder"],
      "LLM": subset.iloc[0]["LLM"],
      "Latency (s)": round(subset["latency"].mean(), 3),
      "Source Attribution": round(subset["source_attribution"].mean(), 2),
      "Faithfulness": _mean("faithfulness"),
      "Context Recall": _mean("context_recall"),
      "Context Precision": _mean("context_precision"),
      "Answer Relevancy": _mean("answer_relevancy"),
    })

    # ── Write leaderboard after each config ───────────────────────────────
    pd.DataFrame(existing_leaderboard).to_csv(leaderboard_file, index=False, sep=";")
    print(f"  Updated leaderboard ({len(existing_leaderboard)} configs so far)")

  print(f"\nSaved {len(existing_pq)} per-question score rows to {per_question_file}")
  print(f"Leaderboard saved to {leaderboard_file}")

  df_leaderboard = pd.DataFrame(existing_leaderboard)
  print("\n--- Leaderboard Summary ---")
  print(df_leaderboard.to_string())


if __name__ == "__main__":
  setup_logging("ragas_eval")
  asyncio.run(evaluate_results())
