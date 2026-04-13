from typing import List, Optional
from haystack import component, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter

# Semantic chunking
@component
class SemanticEmbeddingChunker:
  def __init__(
    self,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentences_per_group: int = 1, # of 2
    percentile: float = 0.95,
    min_length: int = 50,
    max_length: int = 1000,
    device: Optional[str] = None
  ):
    self.embedder = SentenceTransformersDocumentEmbedder(
      model=model_name, 
      device=device
    )
    
    self.splitter = EmbeddingBasedDocumentSplitter(
      document_embedder=self.embedder,
      sentences_per_group=sentences_per_group,
      percentile=percentile,
      min_length=min_length,
      max_length=max_length
    )

  def warm_up(self):
    self.embedder.warm_up()

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    return self.splitter.run(documents=documents)