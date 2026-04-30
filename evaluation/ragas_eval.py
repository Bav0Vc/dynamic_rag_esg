import os
import json
import httpx
import asyncio
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ragas.llms import llm_factory
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

_OLLAMA_BASE_URL = "http://host.docker.internal:11434/v1"
# # From any directory, e.g. C:\Users\vanco\Repos\dynamic_rag_esg
# Set-Content -Path "Modelfile" -Value "FROM gemma3:12b`nPARAMETER num_predict -1`nPARAMETER num_ctx 16384"
# ollama create gemma3-eval:12b -f Modelfile
_OLLAMA_MODEL = "gemma3-eval:12b"


class _TokenTracker:
  def __init__(self):
    self.reset()

  def reset(self):
    self.prompt_tokens = 0
    self.completion_tokens = 0
    self.calls = 0

_token_tracker = _TokenTracker()


async def _on_response(response: httpx.Response) -> None:
  await response.aread()
  try:
    data = json.loads(response.content)
    usage = data.get("usage") or {}
    if usage:
      _token_tracker.prompt_tokens += usage.get("prompt_tokens", 0)
      _token_tracker.completion_tokens += usage.get("completion_tokens", 0)
      _token_tracker.calls += 1
  except Exception:
    pass


# ── Per-sample scorer ─────────────────────────────────────────────────────────

async def score_sample(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, row):
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
    print(f"Faithfulness failed: {type(e).__name__}: {e}")
    print(traceback.format_exc())
    scores["faithfulness"] = np.nan

  try:
    result = await context_recall_m.ascore(
      user_input=user_input,
      retrieved_contexts=retrieved_contexts,
      reference=reference,
    )
    scores["context_recall"] = result.value
  except Exception as e:
    print(f"ContextRecall failed: {type(e).__name__}: {e}")
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
    print(f"ContextPrecision failed: {type(e).__name__}: {e}")
    print(traceback.format_exc())
    scores["context_precision"] = np.nan

  try:
    result = await answer_relevancy_m.ascore(
      user_input=user_input,
      response=response,
    )
    scores["answer_relevancy"] = result.value
  except Exception as e:
    print(f"AnswerRelevancy failed: {type(e).__name__}: {e}")
    print(traceback.format_exc())
    scores["answer_relevancy"] = np.nan

  return scores


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

  http_client = httpx.AsyncClient(event_hooks={"response": [_on_response]})
  llm_client = AsyncOpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama", http_client=http_client)

  evaluator_llm = llm_factory(_OLLAMA_MODEL, provider="openai", client=llm_client)

  # Independent from the three evaluated embedders (bge-m3, arctic-embed, multilingual-e5).
  evaluator_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

  faithfulness_m = Faithfulness(llm=evaluator_llm)
  context_recall_m = ContextRecall(llm=evaluator_llm)
  context_precision_m = ContextPrecision(llm=evaluator_llm)
  answer_relevancy_m = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

  for config in df["Configuration"].unique():
    if config in already_done:
      print(f"Skipping {config} (already evaluated)")
      continue

    current_time_config = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{current_time_config}] | Evaluating: {config}")
    subset = df[df["Configuration"] == config].reset_index(drop=True)

    rows_scores = []
    for _, row in subset.iterrows():
      _token_tracker.reset()
      scores = await score_sample(faithfulness_m, context_recall_m, context_precision_m, answer_relevancy_m, row)
      current_time_question = datetime.now().strftime("%H:%M:%S")
      print(
        f"[{current_time_question}] |   [{row['question_id']}]  "
        f"eval-tokens → prompt: {_token_tracker.prompt_tokens:,}  "
        f"completion: {_token_tracker.completion_tokens:,}  "
        f"calls: {_token_tracker.calls}"
      )
      rows_scores.append(scores)

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
  from logs.logger import setup_logging
  setup_logging("ragas_eval")
  asyncio.run(evaluate_results())
