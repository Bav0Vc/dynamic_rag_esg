from typing import List
import pymupdf4llm
from haystack import component, Document


@component
class PyMuPDF4LLMConverter:
  """
  Converts PDFs to Markdown using pymupdf4llm (backed by PyMuPDF).
  Tables are rendered as Markdown tables; images are described inline.
  Outputs one Document per page. Blank/image-only pages are skipped.
  Metadata keys (file_path, page_number) are compatible with DocumentCleaner.
  """

  @component.output_types(documents=List[Document])
  def run(self, sources: List[str]) -> dict:
    documents = []
    for pdf_path in sources:
      page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
      for chunk in page_chunks:
        text = chunk["text"].strip()
        if not text:
          continue
        documents.append(Document(
          content=text,
          meta={
            "file_path": chunk["metadata"]["file_path"],
            "page_number": chunk["metadata"]["page_number"],
          }
        ))
    return {"documents": documents}
