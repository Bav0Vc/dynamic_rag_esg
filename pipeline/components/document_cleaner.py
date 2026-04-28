import re
from pathlib import Path
from haystack import component, Document

@component
class DocumentCleaner:
  """
  Runs after PyMuPDF4LLMConverter.
  - Normalises whitespace while preserving Markdown table formatting
  - Normalises meta keys so each Document has:
    source  — filename  (used in citation tags)
    page    — page number
  """
  @component.output_types(documents=list[Document])
  def run(self, documents: list[Document]) -> dict:
    cleaned = []
    for doc in documents:
      text = self._clean_text(doc.content or "")
      meta = self._build_meta(doc.meta)
      cleaned.append(Document(content=text, meta=meta))
    return {"documents": cleaned}

  def _clean_text(self, text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

  def _build_meta(self, original_meta: dict) -> dict:
    meta = dict(original_meta)

    if "file_path" in meta:
      meta["source"] = Path(meta.pop("file_path")).name

    if "page_number" in meta:
      meta["page"] = meta.pop("page_number")
    else:
      meta.setdefault("page", "?")

    return meta
