from typing import List
from haystack import component, Document
from haystack.components.preprocessors import RecursiveDocumentSplitter

@component
class RecursiveSplitter:
  def __init__(self, separators: List[str] = ["\n\n", "\n", ".", " ", ""], split_length: int = 150, split_overlap: int = 20):
    self.internal_splitter = RecursiveDocumentSplitter(
      separators=separators,
      split_length=split_length,
      split_overlap=split_overlap,
      split_unit="word",
    )

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    return self.internal_splitter.run(documents=documents)
