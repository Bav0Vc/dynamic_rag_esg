import os
import time
from typing import List
from pathlib import Path
from datetime import datetime
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
from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
# Document Store & Writer
from qdrant_client import QdrantClient
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

load_dotenv()

# Embedding models (1024-dim vectors), via HF Inference API
_EMBEDDER_CONFIGS = [
  {"model_id": "BAAI/bge-m3",                             "api_model": "BAAI/bge-m3"},
  {"model_id": "snowflake/arctic-embed-1-v2.0",           "api_model": "Snowflake/snowflake-arctic-embed-l-v2.0"},
  {"model_id": "intfloat/multilingual-e5-large-instruct", "api_model": "intfloat/multilingual-e5-large"},
]

# Batching requests + delays => counter rate limits
_BATCH_SIZE = 16
_MAX_RETRIES = 8
_RETRY_BASE_DELAY = 15

@component
class RetryingHFDocumentEmbedder:
  """Wraps HuggingFaceAPIDocumentEmbedder with per-batch retry and exponential back-off.

  Splits the document list into batches so a rate-limit error only causes a
  retry of the affected batch, not the entire corpus.
  """

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
            print(
              f"\n[embedder] batch {batch_label} failed"
              f"\n(attempt {attempt + 1}/{self.max_retries}): {exc}. "
              # f"\nRetrying in {wait:.0f}s..."
            )
            # time.sleep(wait)
            remaining_wait = float(wait)
            while remaining_wait > 0:
                print(f"Retrying in {remaining_wait:.0f}s...")
                sleep_interval = min(remaining_wait, 5.0)
                time.sleep(sleep_interval)
                remaining_wait -= sleep_interval
          else:
            print(
              f"\n[embedder] batch {batch_label} permanently failed after {self.max_retries} attempts."
              f"\nLast error: {exc}"
            )
            raise
    return {"documents": embedded}


print("Clearing entire Qdrant database...")
client = QdrantClient(
  url=os.getenv("QDRANT_URL"),
  api_key=os.getenv("QDRANT_API_KEY")
)

collections_res = client.get_collections()
for c in collections_res.collections:
  print(f"Deleting collection: {c.name}")
  client.delete_collection(c.name)
print("Database cleared.")


config_counter=0
for chunker_class in [RecursiveSplitter, FixedSizeWordSplitter, SemanticEmbeddingChunker]:
  for emb_cfg in _EMBEDDER_CONFIGS:
    config_counter += 1
    model_id = emb_cfg["model_id"]
    api_model = emb_cfg["api_model"]

    embedder = RetryingHFDocumentEmbedder(
        api_model=api_model,
        token=Secret.from_env_var("HF_TOKEN"),
    )

    chunker = chunker_class()
    config_name = f"{chunker.__class__.__name__} | {model_id}"
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{current_time}] Indexing config ({config_counter}/9): {config_name}")

    # Collection name must match query_pipeline.py convention: chunker_model_id (slash -> dash)
    collection_name = f"{chunker.__class__.__name__}_{model_id}".replace("/", "-").lower()

    document_store = QdrantDocumentStore(
      url=os.getenv("QDRANT_URL"),
      api_key=Secret.from_env_var("QDRANT_API_KEY"),
      index=collection_name,
      embedding_dim=1024,
      recreate_index=True,
    )

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

    raw_data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    pdf_paths = [str(p) for p in raw_data_dir.glob("*.pdf")]

    if not pdf_paths:
      print("No PDF files found in /data/raw!")
      continue

    indexing_pipe.run({"converter": {"sources": pdf_paths}})
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] Finished indexing for {config_name}")
