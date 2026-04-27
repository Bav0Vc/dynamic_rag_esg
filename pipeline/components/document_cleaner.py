import re
from pathlib import Path
from haystack import component, Document

@component
class DocumentCleaner:
  """
  Runs after PyPDFToDucment
  - Cleans raw PDF text
  - Normalises meta keys so each Document has;
    source   — filename  (used in XML citation tags)
    page     — page number
  """
  @component.output_types(documents=list[Document])
  def run(self, documents: list[Document]) -> dict:
    cleaned = []
    for doc in documents:
      text = self._clean_text(doc.content or "")
      meta = self._build_meta(doc.meta)
      cleaned.append(Document(content=text, meta=meta))
    return {"documents": cleaned}

  # Private helpers
  def _clean_text(self, text: str) -> str:
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs to a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Remove lines that are only a number
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove lines that are only dashes/underscores
    text = re.sub(r"^\s*[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()
  
  def _build_meta(self, original_meta: dict) -> dict:
    meta = dict(original_meta)

    # Standardize 'page' from 'page_number'
    if "page_number" in meta:
        meta["page"] = meta.pop("page_number")
    else:
        meta.setdefault("page", "?")

    # Standardize 'source' from 'file_path'
    if "file_path" in meta:
        meta["source"] = Path(meta.pop("file_path")).name
    
    return meta