from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from components.document_cleaner import DocumentCleaner

indexing = Pipeline()
indexing.add_component("converter", PyPDFToDocument())
indexing.add_component("cleaner",   DocumentCleaner())

indexing.connect("converter.documents", "cleaner.documents")