import os
import sys
import asyncio
from evaluation.ragas_eval import evaluate_results
from pipeline.indexing_pipeline import run_indexing
from orchestration.benchmark_loop import run_benchmark


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

# Add pipeline dir to path => correctly importing siblings of indexing_pipeline
_PIPELINE_DIR = os.path.join(_PROJECT_ROOT, "pipeline")
if _PIPELINE_DIR not in sys.path:
  sys.path.insert(0, _PIPELINE_DIR)


# ── Step 1: Index all chunker × embedder combinations ────────────────────────
print("=" * 60)
print("STEP 1: Indexing pipeline")
print("=" * 60)
run_indexing()

# ── Step 2: Run query benchmark over all configurations ──────────────────────
print("\n" + "=" * 60)
print("STEP 2: Benchmark loop")
print("=" * 60)
run_benchmark()

# ── Step 3: RAGAS evaluation ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: RAGAS evaluation")
print("=" * 60)
asyncio.run(evaluate_results())
