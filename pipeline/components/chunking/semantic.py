from typing import List, Optional
from haystack import component, Document
from transformers import logging as hf_logging
from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

@component
class SemanticEmbeddingChunker:
  def __init__(
    self,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    sentences_per_group: int = 3,
    percentile: float = 0.95,
    min_length: int = 50,
    max_length: int = 250,
    device: Optional[str] = None
  ):
    hf_logging.set_verbosity_error()

    self.embedder = SentenceTransformersDocumentEmbedder(
      model=model_name, 
      device=device
    )
    
    self.splitter = EmbeddingBasedDocumentSplitter(
      document_embedder=self.embedder,
      sentences_per_group=sentences_per_group,
      percentile=percentile,
      min_length=min_length,
      max_length=max_length,
    )

  # For GPU
  def warm_up(self):
    self.embedder.warm_up()

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    return self.splitter.run(documents=documents)