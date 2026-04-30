import os
import sys
import asyncio
from itertools import product
from dotenv import load_dotenv
from logs.logger import setup_logging
from qdrant_client import QdrantClient
from evaluation.ragas_eval import evaluate_results
from orchestration.benchmark_loop import run_benchmark
from pipeline.indexing_pipeline import run_indexing, _CHUNKERS, _EMBEDDERS


load_dotenv()
setup_logging("run_pipeline")


def _check_existing_indexing_state() -> int | None:
  """
  Inspects Qdrant for existing collections from a previous indexing run.

  Returns the combination index to start from:
    - 0           → fresh start (wipe everything and reindex all)
    - N (N > 0)   → resume from combination N (skip the first N which are complete)
    - None        → skip indexing entirely (user cancelled)

  Completeness of the last existing collection is inferred by comparing its
  chunk count against same-chunker peers: same chunker always produces the
  same number of chunks regardless of embedder, so a significantly lower count
  indicates the collection was interrupted mid-run.
  """
  

  combinations = list(product(_CHUNKERS, _EMBEDDERS))
  n_total = len(combinations)
  expected_names = [f"{c}_{e}".replace("/", "-").lower() for c, e in combinations]

  try:
    client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    existing = {c.name for c in client.get_collections().collections}
  except Exception as exc:
    print(f"  [warning] Could not connect to Qdrant to check existing collections: {exc}")
    print("  Proceeding with a fresh start.")
    return 0

  # Indices of collections that already exist (in pipeline order)
  existing_indices = [i for i, name in enumerate(expected_names) if name in existing]

  if not existing_indices:
    return 0  # nothing in Qdrant yet = clean start

  last_idx = existing_indices[-1]
  last_name = expected_names[last_idx]
  last_count = client.get_collection(last_name).points_count

  # Determine if last collection is complete
  # Same chunker -> same number of chunks, so compare against peer counts
  last_chunker = combinations[last_idx][0]
  peer_counts = [
    client.get_collection(expected_names[i]).points_count
    for i in existing_indices[:-1]
    if combinations[i][0] == last_chunker
  ]

  if peer_counts:
    avg_peer = sum(peer_counts) / len(peer_counts)
    last_is_complete = last_count >= 0.9 * avg_peer
  else:
    # No same-chunker peers to compare against — assume incomplete to be safe
    last_is_complete = False

  resume_idx = last_idx + 1 if last_is_complete else last_idx
  n_complete = len(existing_indices) - (0 if last_is_complete else 1)

  # ── Display current state ─────────────────────────────────────────────────
  print(f"\n  Found {len(existing_indices)} existing collection(s) out of {n_total}:")
  for i in existing_indices[:-1]:
    chunker, embedder = combinations[i]
    print(f"    [complete]   ({i + 1:>2}/{n_total}) {chunker} | {embedder}")
  last_chunker_name, last_embedder = combinations[last_idx]
  last_label = "complete  " if last_is_complete else "INCOMPLETE"
  print(f"    [{last_label}] ({last_idx + 1:>2}/{n_total}) {last_chunker_name} | {last_embedder}  ({last_count} chunks)")

  # ── All collections are complete ─────────────────────────────────────────
  if resume_idx >= n_total:
    print(f"\n  All {n_total} collections are already complete.")
    ans = input("  Rerun everything from scratch? [Y/n]: ").strip().upper()
    if ans == "Y":
      return 0
    return None  # skip indexing

  # ── Partial completion — offer to continue ────────────────────────────────
  next_chunker, next_embedder = combinations[resume_idx]
  print(f"\n  {n_complete} collection(s) complete.")
  print(f"  Will resume from ({resume_idx + 1}/{n_total}): {next_chunker} | {next_embedder}")
  if not last_is_complete:
    print(f"  (The last collection was incomplete and will be re-indexed from scratch.)")
  ans = input("  [C]ontinue from here / [R]erun everything from scratch / any other key to cancel: ").strip().upper()
  if ans == "C":
    return resume_idx
  if ans == "R":
    return 0
  return None  # cancel


print("Starting pipeline:")

indexing = input("(Re)run indexing pipeline? [Y/n]: ").strip().upper()
if indexing == "Y":
  # ── Step 1: Index all chunker × embedder combinations ────────────────────────
  print("=" * 60)
  print("STEP 1: Indexing pipeline")
  print("=" * 60)
  resume_from = _check_existing_indexing_state()
  if resume_from is not None:
    run_indexing(resume_from=resume_from)

benchmarking = input("(Re)run benchmark loop? [Y/n]: ").strip().upper()
if benchmarking == "Y":
  # ── Step 2: Run query benchmark over all configurations ──────────────────────
  print("\n" + "=" * 60)
  print("STEP 2: Benchmark loop")
  print("=" * 60)
  run_benchmark()

evaluating = input("(Re)run RAGAS evaluation pipeline? [Y/n]: ").strip().upper()
if evaluating == "Y":
  # ── Step 3: RAGAS evaluation ─────────────────────────────────────────────────
  print("\n" + "=" * 60)
  print("STEP 3: RAGAS evaluation")
  print("=" * 60)
  asyncio.run(evaluate_results())

sys.exit()
