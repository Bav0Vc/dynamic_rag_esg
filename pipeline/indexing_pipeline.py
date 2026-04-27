import os
import sys
import time
from typing import List
from pathlib import Path
from datetime import datetime
from itertools import product
from dotenv import load_dotenv
from haystack.utils import Secret
from haystack import Pipeline, component, Document
# Cleaner & Converter
from components.document_cleaner import DocumentCleaner
from haystack.components.converters import PyPDFToDocument
# Chunking
from components.chunking.fixed import FixedSizeWordSplitter
from components.chunking.recursive import RecursiveSplitter
from components.chunking.semantic import SemanticEmbeddingChunker
# Embedding
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder, SentenceTransformersDocumentEmbedder
# Document Store & Writer
from qdrant_client import QdrantClient
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
  sys.path.insert(0, _PIPELINE_DIR)

_PROJECT_ROOT = os.path.dirname(_PIPELINE_DIR)
if _PROJECT_ROOT not in sys.path:
  sys.path.insert(0, _PROJECT_ROOT)

from config.hypster_config import pipeline_config
from hypster import instantiate

load_dotenv()

_BATCH_SIZE = 16
_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 15

_CHUNKERS = ["RecursiveSplitter", "FixedSizeWordSplitter", "SemanticEmbeddingChunker"]
_EMBEDDERS = ["BAAI/bge-m3", "snowflake/arctic-embed-1-v2.0", "intfloat/multilingual-e5-large-instruct"]


@component
class RetryingHFDocumentEmbedder:
  """Wraps HuggingFaceAPIDocumentEmbedder with per-batch retry and exponential back-off."""

  def __init__(self, api_model: str, token: Secret, batch_size: int = _BATCH_SIZE, max_retries: int = _MAX_RETRIES, retry_base_delay: float = _RETRY_BASE_DELAY):
    self._inner = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api", api_params={"model": api_model}, token=token, truncate=None, normalize=None)
    self.batch_size = batch_size
    self.max_retries = max_retries
    self.retry_base_delay = retry_base_delay

  def warm_up(self):
    if hasattr(self._inner, "warm_up"):
      self._inner.warm_up()

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    embedded: List[Document] = []
    total = len(documents)
    for start in range(0, total, self.batch_size):
      batch = documents[start:start + self.batch_size]
      batch_label = f"docs {start + 1}–{min(start + self.batch_size, total)} / {total}"
      for attempt in range(self.max_retries):
        try:
          result = self._inner.run(documents=batch)
          embedded.extend(result["documents"])
          break
        except Exception as exc:
          if attempt < self.max_retries - 1:
            wait = self.retry_base_delay * (2 ** attempt)
            print(f"\n[embedder] batch {batch_label} failed (attempt {attempt + 1}/{self.max_retries})\nError: {exc}.")
            remaining_wait = float(wait)
            while remaining_wait > 0:
              print(f"Retrying in {remaining_wait:.0f}s...")
              sleep_interval = min(remaining_wait, 5.0)
              time.sleep(sleep_interval)
              remaining_wait -= sleep_interval
          else:
            print(f"\n[embedder] batch {batch_label} permanently failed after {self.max_retries} attempts.\nLast error: {exc}")
            raise
    return {"documents": embedded}


def _make_chunker(chunker_name: str):
  if chunker_name == "RecursiveSplitter":
    return RecursiveSplitter()
  if chunker_name == "FixedSizeWordSplitter":
    return FixedSizeWordSplitter()
  if chunker_name == "SemanticEmbeddingChunker":
    return SemanticEmbeddingChunker()
  raise ValueError(f"Unknown chunker: {chunker_name}")


def _make_doc_embedder(emb_cfg: dict):
  if emb_cfg["backend"] == "sentence-transformers":
    return SentenceTransformersDocumentEmbedder(model=emb_cfg["api_model"])
  return RetryingHFDocumentEmbedder(
    api_model=emb_cfg["api_model"],
    token=Secret.from_env_var("HF_TOKEN"),
  )


def run_indexing() -> None:
  print("Clearing entire Qdrant database...")
  client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
  for c in client.get_collections().collections:
    print(f"  Deleting collection: {c.name}")
    client.delete_collection(c.name)
  print("Database cleared.\n")

  combinations = list(product(_CHUNKERS, _EMBEDDERS))
  config_counter = 0

  for chunker_name, embedder_model in combinations:
    config_counter += 1
    overrides = {
      "chunking.chunker_name": chunker_name,
      "embedding.model": embedder_model,
      "llm.name": "Qwen-2.5-14B",  # LLM unused during indexing; keep a valid default
    }
    config = instantiate(pipeline_config, values=overrides, on_unknown="raise")
    emb_cfg = config["embedding"]

    config_name = f"{chunker_name} | {embedder_model}"
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Indexing config ({config_counter}/{len(combinations)}): {config_name}")

    collection_name = f"{chunker_name}_{embedder_model}".replace("/", "-").lower()

    document_store = QdrantDocumentStore(
      url=os.getenv("QDRANT_URL"),
      api_key=Secret.from_env_var("QDRANT_API_KEY"),
      index=collection_name,
      embedding_dim=emb_cfg["dims"],
      recreate_index=True,
    )

    chunker = _make_chunker(chunker_name)
    embedder = _make_doc_embedder(emb_cfg)

    indexing_pipe = Pipeline()
    indexing_pipe.add_component("converter", PyPDFToDocument())
    indexing_pipe.add_component("cleaner", DocumentCleaner())
    indexing_pipe.add_component("chunker", chunker)
    indexing_pipe.add_component("embedder", embedder)
    indexing_pipe.add_component("writer", DocumentWriter(document_store=document_store))

    indexing_pipe.connect("converter.documents", "cleaner.documents")
    indexing_pipe.connect("cleaner.documents", "chunker.documents")
    indexing_pipe.connect("chunker.documents", "embedder.documents")
    indexing_pipe.connect("embedder.documents", "writer.documents")

    raw_data_dir = Path(_PIPELINE_DIR).parent / "data" / "raw"
    pdf_paths = [str(p) for p in raw_data_dir.glob("*.pdf")]

    if not pdf_paths:
      print("  No PDF files found in /data/raw!")
      continue

    indexing_pipe.run({"converter": {"sources": pdf_paths}})
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Finished indexing for {config_name}")


if __name__ == "__main__":
  run_indexing()
