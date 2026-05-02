import re
from pathlib import Path
from haystack import component, Document


@component
class ChunkMetaCleaner:
  """Strips internal Haystack splitter fields (page_number) from chunk metadata after chunking."""
  @component.output_types(documents=list[Document])
  def run(self, documents: list[Document]) -> dict:
    for doc in documents:
      doc.meta.pop("page_number", None)
    return {"documents": documents}


@component
class DocumentCleaner:
  """
  Runs after UnstructuredFileConverter.
  - Drops empty documents
  - Normalises whitespace while preserving Markdown table formatting
  - Normalises meta keys so each Document has:
      source  — filename           (used in citation tags)
      page    — page number (PDF/DOCX) or sheet name (XLSX)
  """
  @component.output_types(documents=list[Document])
  def run(self, documents: list[Document]) -> dict:
    cleaned = []
    for doc in documents:
      text = self._clean_text(doc.content or "")
      if not text:
        continue
      meta = self._build_meta(doc.meta)
      cleaned.append(Document(content=text, meta=meta))
    return {"documents": cleaned}

  def _clean_text(self, text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

  def _build_meta(self, original_meta: dict) -> dict:
    meta = dict(original_meta)

    # Unstructured provides 'filename' (bare name)
    # Integration also adds 'file_path' (full path)
    # Prefer 'filename' since it needs no further processing
    if "filename" in meta:
      meta["source"] = meta.pop("filename")
    elif "file_path" in meta:
      meta["source"] = Path(meta["file_path"]).name
    meta.pop("file_path", None)

    # For Excel files use the sheet name as the page identifier
    # For everything else use the numeric page_number supplied by Unstructured
    source = meta.get("source", "")
    if source.lower().endswith((".xlsx", ".xls")) and "page_name" in meta:
      meta["page"] = meta.pop("page_name")
    elif "page_number" in meta:
      meta["page"] = meta.pop("page_number")
    else:
      meta.setdefault("page", "?")

    return meta
