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
from haystack.components.embedders import OpenAIDocumentEmbedder, SentenceTransformersDocumentEmbedder
# Document Store & Writer
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient
from haystack.components.writers import DocumentWriter

load_dotenv()

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
  for embedder_config in [
    {"class": SentenceTransformersDocumentEmbedder, "kwargs": {"model": "BAAI/bge-m3"}},
    #{"class": OpenAIDocumentEmbedder, "kwargs": {"model": "text-embedding-3-large"}},
    {"class": SentenceTransformersDocumentEmbedder, "kwargs": {"model": "intfloat/multilingual-e5-large-instruct"}}
  ]:
    
    chunker = chunker_class()
    embedder = embedder_config["class"](**embedder_config["kwargs"])
    
    config_name = f"{chunker.__class__.__name__} | {embedder.model}"
    print(f"Indexing config: {config_name}")

    collection_name = f"{chunker.__class__.__name__}_{embedder.model}".replace("/", "-").lower()

    # Determine embedding dimension based on the model
    if "bge-m3" in embedder.model:
      dim = 1024
    elif "text-embedding-3-large" in embedder.model:
      dim = 3072
    elif "multilingual-e5-large-instruct" in embedder.model:
      dim = 1024
    else:
      dim = 768

    document_store = QdrantDocumentStore(
      url=os.getenv("QDRANT_URL"),
      api_key=Secret.from_env_var("QDRANT_API_KEY"),
      index=collection_name,
      embedding_dim=dim,
      recreate_index=True
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

    raw_data_dir = Path("data/raw")
    pdf_paths = [str(p) for p in raw_data_dir.glob("*.pdf")]
    
    if not pdf_paths:
      print("No PDF files found in /data/raw!")
      continue

    indexing_pipe.run({"converter": {"sources": pdf_paths}})
    print(f"Finished indexing for {config_name}")