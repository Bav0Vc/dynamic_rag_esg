from typing import List
from haystack import component, Document

# Recursive character text splitting
@component
class RecursiveSplitter:
  def __init__(self, separators: List[str] = ["\n\n", "\n", ".", " ", ""], split_length: int = 150, split_overlap: int = 20):
    self.separators = separators
    self.split_length = split_length
    self.split_overlap = split_overlap

  @component.output_types(documents=List[Document])
  def run(self, documents: List[Document]):
    final_docs = []
    for doc in documents:
      text = doc.content
      chunks = self._recursive_split(text, self.separators)

      search_from = 0
      for i, chunk in enumerate(chunks):
        pos = text.find(chunk, search_from)
        if pos >= 0:
          end_pos = pos + len(chunk)
          page_start = text[:pos].count("\f") + 1
          page_end = text[:end_pos].count("\f") + 1
          page = page_start if page_start == page_end else f"{page_start}-{page_end}"
          search_from = pos
        else:
          page = doc.meta.get("page", "?")

        final_docs.append(Document(content=chunk, meta={**doc.meta, "page": page, "chunk_id": i}))

    return {"documents": final_docs}

  def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
    if not separators:
      return [text]

    sep = separators[0]
    # Need to properly split and then recurse if chunks are still too big
    parts = text.split(sep)
    
    final_chunks = []
    current_chunk = ""
    
    # When text is on multiple pages (returns range of pages instead of 1 page number)
    for part in parts:
      # If word count > split_length, recurse with next separator
      if len(part.split()) > self.split_length and len(separators) > 1:
        sub_chunks = self._recursive_split(part, separators[1:])
        final_chunks.extend(sub_chunks)
      else:
        # Build up chunks until they reach split_length WORDS
        if len((current_chunk + sep + part).split()) <= self.split_length:
          current_chunk = current_chunk + sep + part if current_chunk else part
        else:
          if current_chunk:
            final_chunks.append(current_chunk.strip())
          current_chunk = part
          
    if current_chunk:
      final_chunks.append(current_chunk.strip())

    return [p for p in final_chunks if p.strip()]