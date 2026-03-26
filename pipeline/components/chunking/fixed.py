from typing import List
from haystack import component, Document
from haystack.components.preprocessors import DocumentSplitter

# Fixed size chunking
@component
class FixedSizeWordSplitter:
  def __init__(self, split_length: int = 150, split_overlap: int = 20):
    self.internal_splitter = DocumentSplitter(
      split_by="word", 
      split_length=split_length, 
      split_overlap=split_overlap
    )

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    return self.internal_splitter.run(documents=documents)