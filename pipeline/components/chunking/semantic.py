import dataclasses
from typing import List
from haystack import component, Document
from transformers import logging as hf_logging
from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter, RecursiveDocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder


@component
class SemanticEmbeddingChunker:
  def __init__(
    self,
    max_length: int,
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    sentences_per_group: int = 2,
    percentile: float = 0.95,
    min_length: int = 50,
    language: str = "it",
  ):
    hf_logging.set_verbosity_error()
    self.max_length = max_length

    self.embedder = SentenceTransformersDocumentEmbedder(
      model=model_name,
      progress_bar=False,
    )
    self.splitter = EmbeddingBasedDocumentSplitter(
      document_embedder=self.embedder,
      sentences_per_group=sentences_per_group,
      percentile=percentile,
      min_length=min_length,
      max_length=max_length,
      language=language,
    )

    overlap_chars = int(self.max_length * 0.15)

    self._fallback = RecursiveDocumentSplitter(
      separators=["\n\n", "\n", ". ", " "],
      split_length=self.max_length,
      split_overlap=overlap_chars,
      split_unit="char"
    )

  def warm_up(self) -> None:
    self.splitter.warm_up()

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]) -> dict:
    chunks = self.splitter.run(documents=documents)["documents"]
    final: List[Document] = []

    for doc in chunks:
      if len(doc.content or "") > self.max_length:
        sub_chunks = self._fallback.run(documents=[doc])["documents"]
        for sub in sub_chunks:
          final.append(dataclasses.replace(sub, meta={**doc.meta, **sub.meta}))
      else:
        final.append(doc)

    return {"documents": final}


