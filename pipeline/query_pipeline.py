import os
import pandas as pd
from haystack import Pipeline
from dotenv import load_dotenv
# Embedding
from haystack.components.embedders import OpenAIDocumentEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.embedders import OpenAITextEmbedder, SentenceTransformersTextEmbedder
# Chunking
from components.chunking.fixed import FixedSizeWordSplitter
from components.chunking.recursive import RecursiveSplitter
from components.chunking.semantic import SemanticEmbeddingChunker
# Document Store & Retriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.utils import Secret
# Prompts & Generators
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

load_dotenv()

chunking_strategies = [
  RecursiveSplitter(),
  FixedSizeWordSplitter(),
  SemanticEmbeddingChunker()
]

embedding_models = [
  OpenAIDocumentEmbedder(model="text-embedding-3-large"),
  SentenceTransformersDocumentEmbedder(model="BAAI/bge-m3"),
  SentenceTransformersDocumentEmbedder(model="intfloat/multilingual-e5-large-instruct")
]

llm_models = [
  {
    "name": "GPT-4o-mini",
    "instance": OpenAIGenerator(model="gpt-4o-mini")
  },
  {
    "name": "Gemini-2.5-Flash",
    "instance": GoogleGenAIChatGenerator(model="gemini-2.5-flash")
  },
  {
    "name": "Mistral-Large-2",
    "instance": MistralChatGenerator(model="mistral-large-2407")
  }
]

question = "Wat is de naam en het adres van de school van Bavo Vancoppernolle?"
test_results = []

for chunker_class in [RecursiveSplitter, FixedSizeWordSplitter, SemanticEmbeddingChunker]:
  for embedder_config in [
    {"class": SentenceTransformersDocumentEmbedder, "kwargs": {"model": "BAAI/bge-m3"}, "text_class": SentenceTransformersTextEmbedder},
    {"class": OpenAIDocumentEmbedder, "kwargs": {"model": "text-embedding-3-large"}, "text_class": OpenAITextEmbedder},
    {"class": SentenceTransformersDocumentEmbedder, "kwargs": {"model": "intfloat/multilingual-e5-large-instruct"}, "text_class": SentenceTransformersTextEmbedder}
  ]:
    for llm_config in llm_models:
      chunker = chunker_class()
      embedder_model_name = embedder_config["kwargs"]["model"]
      
      config_name = f"{chunker.__class__.__name__} | {embedder_model_name} | {llm_config['name']}"
      print(f"Running config: {config_name}")

      collection_name = f"{chunker.__class__.__name__}_{embedder_model_name}".replace("/", "-").lower()

      if "bge-m3" in embedder_model_name:
        dim = 1024
      elif "text-embedding-3-large" in embedder_model_name:
        dim = 3072
      elif "multilingual-e5-large-instruct" in embedder_model_name:
        dim = 1024
      else:
        dim = 768

      document_store = QdrantDocumentStore(
        url=os.getenv("QDRANT_URL"),
        api_key=Secret.from_env_var("QDRANT_API_KEY"),
        index=collection_name,
        embedding_dim=dim,
        recreate_index=False 
      )

      try:
        out = document_store.count_documents()
        if out == 0:
          print(f"  -> Skipping. Collection {collection_name} is empty. Run indexing first.")
          continue
      except Exception as e:
        print(f"  -> Could not connect or find collection {collection_name}. Skipping. {e}")
        continue

      # Query embedding
      text_embedder = embedder_config["text_class"](model=embedder_model_name)

      if llm_config["name"] == "GPT-4o-mini":
        llm_instance = OpenAIGenerator(model="gpt-4o-mini")
      elif llm_config["name"] == "Gemini-2.5-Flash":
        llm_instance = GoogleGenAIChatGenerator(model="gemini-2.5-flash")
      else:
        llm_instance = MistralChatGenerator(model="mistral-large-2407")

      query_pipe = Pipeline()
      query_pipe.add_component("text_embedder", text_embedder)
      query_pipe.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store))
      query_pipe.add_component("prompt_builder", PromptBuilder(template="""Beantwoord de vraag op basis van de context. Geef ook de filename van het document waarin de opgehaalde chunk staat, op een newline na het antwoord op de vraag.
      Context: 
      {% for doc in documents %} 
        Bestand: {{ doc.meta['file_path'] }}
        Inhoud: {{ doc.content }} 
      {% endfor %}
      Vraag: {{question}}"""))
      query_pipe.add_component("llm", llm_instance)

      query_pipe.connect("text_embedder.embedding", "retriever.query_embedding")
      query_pipe.connect("retriever.documents", "prompt_builder.documents")
      query_pipe.connect("prompt_builder", "llm")

      try:
        response = query_pipe.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
          },
          include_outputs_from={"retriever"}
        )

        reply = response["llm"]["replies"][0]
        final_answer = reply.content if hasattr(reply, 'content') else reply
        # retrieved_documents = [doc.content for doc in response["retriever"]["documents"]]

        test_results.append({
          "Configuration": config_name,
          "Chunker": chunker.__class__.__name__,
          "Embedder": embedder_model_name,
          "LLM": llm_config["name"],
          "Answer": final_answer,
          # "contexts": retrieved_documents
        })
      except Exception as e:
        print(f"Error querying {config_name}: {e}")

pd.DataFrame(test_results).to_csv("./evaluation/results/rag_benchmark_results.csv", index=False)
print("Querying complete. Results saved to rag_benchmark_results.csv")