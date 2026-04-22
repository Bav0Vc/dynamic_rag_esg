import os
from haystack.utils import Secret
from haystack import Pipeline
from dotenv import load_dotenv
from pathlib import Path
# Cleaner & Converter
from components.document_cleaner import DocumentCleaner
from haystack.components.converters import PyPDFToDocument
# Chunking
from components.chunking.fixed import FixedSizeWordSplitter
from components.chunking.recursive import RecursiveSplitter
from components.chunking.semantic import SemanticEmbeddingChunker
# Embedding
from haystack.components.embedders import OpenAIDocumentEmbedder, HuggingFaceAPIDocumentEmbedder
# Document Store & Writer
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient
from haystack.components.writers import DocumentWriter

load_dotenv()

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# All three embedding models output 1024-dimensional vectors
_EMBEDDER_CONFIGS = [
  {"model_id": "BAAI/bge-m3",                           "api_model": "baai/bge-m3",                           "backend": "nvidia"},
  {"model_id": "snowflake/arctic-embed-1-v2.0",          "api_model": "Snowflake/snowflake-arctic-embed-l-v2.0", "backend": "hf"},
  {"model_id": "intfloat/multilingual-e5-large-instruct", "api_model": "intfloat/multilingual-e5-large",          "backend": "hf"},
]

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


for chunker_class in [RecursiveSplitter, FixedSizeWordSplitter, SemanticEmbeddingChunker]:
  for emb_cfg in _EMBEDDER_CONFIGS:
    model_id = emb_cfg["model_id"]
    api_model = emb_cfg["api_model"]
    backend = emb_cfg["backend"]

    if backend == "nvidia":
      embedder = OpenAIDocumentEmbedder(
        model=api_model,
        api_key=Secret.from_env_var("NVIDIA_API_KEY"),
        api_base_url=_NVIDIA_BASE_URL,
      )
    else:  # hf
      embedder = HuggingFaceAPIDocumentEmbedder(
        api_type="serverless_inference_api",
        api_params={"model": api_model},
        token=Secret.from_env_var("HF_TOKEN"),
      )

    chunker = chunker_class()
    config_name = f"{chunker.__class__.__name__} | {model_id}"
    print(f"Indexing config: {config_name}")

    # Collection name must match query_pipeline.py convention: chunker_model_id (slashes → dashes)
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
    print(f"Finished indexing for {config_name}")
