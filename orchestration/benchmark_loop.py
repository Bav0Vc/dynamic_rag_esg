import json
import os
import sys
from itertools import product

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from config.hypster_config import pipeline_config
from hypster import instantiate
from pipeline.query_pipeline import run_query_pipeline

# 
# Configuration space axes
# 
CHUNKERS = ["RecursiveSplitter", "FixedSizeWordSplitter", "SemanticEmbeddingChunker"]
EMBEDDERS = ["BAAI/bge-m3", "snowflake/arctic-embed-1-v2.0", "intfloat/multilingual-e5-large-instruct"]
LLMS = ["Qwen-2.5-14B", "Llama-3.3-70B", "Mistral-Large-2"]

#
# Output paths
# 
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "evaluation", "results")
EVAL_DATASET_PATH = os.path.join(_RESULTS_DIR, "evaluation_dataset.json")
BENCHMARK_CSV_PATH = os.path.join(_RESULTS_DIR, "rag_benchmark_results.csv")


def load_golden_dataset() -> list:
  # path = os.path.join(_PROJECT_ROOT, "data", "golden_set", "example.json")
  path = os.path.join(_PROJECT_ROOT, "data", "golden_set", "vsme_esg.json")
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def run_benchmark() -> None:
  os.makedirs(_RESULTS_DIR, exist_ok=True)

  golden_dataset = load_golden_dataset()
  combinations = list(product(CHUNKERS, EMBEDDERS, LLMS))
  print(f"Starting benchmark: {len(combinations)} configurations × {len(golden_dataset)} questions.\n")

  all_results: list[dict] = []

  for idx, (chunker_name, embedder_model, llm_name) in enumerate(combinations, start=1):
    overrides = {
      "chunking.chunker_name": chunker_name,
      "embedding.model": embedder_model,
      "llm.name": llm_name,
    }
    # Instantiate config
    config = instantiate(pipeline_config, values=overrides, on_unknown="raise")

    print(f"[{idx}/{len(combinations)}] {chunker_name} | {embedder_model} | {llm_name}")
    results = run_query_pipeline(config, golden_dataset)
    all_results.extend(results)

  # Save raw evaluation dataset
  with open(EVAL_DATASET_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
  print(f"\nSaved {len(all_results)} rows to {EVAL_DATASET_PATH}")

  # Save to dataFrame CSV
  df = pd.DataFrame(all_results)
  df.to_csv(BENCHMARK_CSV_PATH, index=False, sep=";")
  print(f"Saved benchmark CSV to {BENCHMARK_CSV_PATH}")


if __name__ == "__main__":
  run_benchmark()
