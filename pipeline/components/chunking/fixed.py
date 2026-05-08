from typing import List
from haystack import component, Document
from haystack.components.preprocessors import RecursiveDocumentSplitter

@component
class FixedSizeTokenSplitter:
  def __init__(self, split_length: int = 400, split_overlap: int = 50):
    # Uses the same token-based splitter as RecursiveCharacterSplitter so both
    # operate in the same unit (tokens). Separators [" ", ""] mean it only splits
    # at word boundaries — no paragraph/sentence awareness — giving fixed-size
    # token windows. This isolates the structural splitting strategy as the variable
    # being benchmarked, not the chunk size or unit.
    self.internal_splitter = RecursiveDocumentSplitter(
      separators=[" ", ""],
      split_length=split_length,
      split_overlap=split_overlap,
      split_unit="token",
    )

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    return self.internal_splitter.run(documents=documents)
