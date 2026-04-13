import re
from pathlib import Path
from haystack import component, Document

@component
class DocumentCleaner:
  """
  Runs after PyPDFToDucment
  - Cleans raw PDF text
  - Extracts structured metadata from a filename pattern: {company}_{year}_{type}.pdf
  - Normalises meta keys so each Document has;
    source   — filename  (used in XML citation tags)
    page     — page number
    year     — reporting year (int)
    company  — company name
    doc_type — report type (ecovadis, vsme, …)
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
    meta = dict(original_meta)  # never mutate original

    # Normalise page key
    meta["page"] = meta.get("page_number", "?")

    # Extract structured fields from filename
    file_path = meta.get("file_path", "")
    filename = Path(file_path).name

    meta["source"] = filename
    extracted = self._parse_filename(filename)
    meta.update(extracted)

    return meta
  
  def _parse_filename(self, filename: str) -> dict:
    # Expected pattern: {company}_{year}_{type}.pdf
    pattern = r"^(?P<company>[^_]+)_(?P<year>\d{4})_(?P<doc_type>.+?)\.pdf$"
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
      return {
        "company": match.group("company"),
        "year": int(match.group("year")),
        "doc_type": match.group("doc_type"),
      }
    # Filename doesn't match the convention
    return {"company": "unknown", "year": None, "doc_type": "unknown"}