import os
import sys
import time

from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.components.generators.mistral import MistralChatGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

load_dotenv()

_PROMPT_TEMPLATE = """Beantwoord de vraag op basis van de context. \
Geef ook de filename van het document waarin de opgehaalde chunk staat, op een newline na het antwoord op de vraag.
Context:
{% for doc in documents %}
  Bestand: {{ doc.meta['file_path'] }}
  Inhoud: {{ doc.content }}
{% endfor %}
Vraag: {{question}}"""

_EMBEDDING_DIMS = {
  "BAAI/bge-m3": 1024,
  "text-embedding-3-large": 3072,
  "intfloat/multilingual-e5-large-instruct": 1024,
}


def _extract_reply_text(reply) -> str:
  if hasattr(reply, "content"):
    if isinstance(reply.content, list):
      return "\n".join(
        c.text if getattr(c, "text", None) is not None else str(c)
        for c in reply.content
      )
    return reply.content if isinstance(reply.content, str) else str(reply.content)
  if hasattr(reply, "text"):
    return reply.text
  return str(reply)


def _build_text_embedder(embedder_model: str):
  if embedder_model == "text-embedding-3-large":
    return OpenAITextEmbedder(model=embedder_model)
  return SentenceTransformersTextEmbedder(model=embedder_model)


def _build_llm(llm_name: str):
  if llm_name == "GPT-4o-mini":
    return OpenAIGenerator(model="gpt-4o-mini")
  if llm_name == "Gemini-2.5-Flash":
    return GoogleGenAIChatGenerator(model="gemini-2.5-flash")
  if llm_name == "Mistral-Large-2":
    return MistralChatGenerator(model="mistral-large-2407")
  raise ValueError(f"Unknown LLM: {llm_name!r}")


def run_query_pipeline(config: dict, golden_dataset: list) -> list:
  """Run a single RAG configuration against the golden dataset.

  Args:
      config: Instantiated Hypster config dict with keys
              ``chunking``, ``embedding``, and ``llm``.
      golden_dataset: List of dicts, each with keys ``id``, ``question``,
                      ``ground_truth``, and ``expected_source``.

  Returns:
      List of result dicts — one entry per (question, config) pair.
  """
  chunker_name: str = config["chunking"]["chunker_name"]
  embedder_model: str = config["embedding"]["model"]
  llm_name: str = config["llm"]["name"]

  config_label = f"{chunker_name} | {embedder_model} | {llm_name}"
  print(f"Running config: {config_label}")

  collection_name = f"{chunker_name}_{embedder_model}".replace("/", "-").lower()
  dim = _EMBEDDING_DIMS.get(embedder_model, 768)

  document_store = QdrantDocumentStore(
    url=os.getenv("QDRANT_URL"),
    api_key=Secret.from_env_var("QDRANT_API_KEY"),
    index=collection_name,
    embedding_dim=dim,
    recreate_index=False,
  )

  try:
    count = document_store.count_documents()
    if count == 0:
      print(f"  -> Skipping. Collection '{collection_name}' is empty.")
      return []
  except Exception as exc:
    print(f"  -> Could not connect to collection '{collection_name}'. Skipping. {exc}")
    return []

  text_embedder = _build_text_embedder(embedder_model)
  llm_instance = _build_llm(llm_name)

  query_pipe = Pipeline()
  query_pipe.add_component("text_embedder", text_embedder)
  query_pipe.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store))
  query_pipe.add_component("prompt_builder", PromptBuilder(template=_PROMPT_TEMPLATE))
  query_pipe.add_component("llm", llm_instance)

  query_pipe.connect("text_embedder.embedding", "retriever.query_embedding")
  query_pipe.connect("retriever.documents", "prompt_builder.documents")
  query_pipe.connect("prompt_builder", "llm")

  results = []
  for item in golden_dataset:
    q = item["question"]
    try:
      start_time = time.time()
      response = query_pipe.run(
        {"text_embedder": {"text": q}, "prompt_builder": {"question": q}},
        include_outputs_from={"retriever"},
      )
      latency = time.time() - start_time

      reply = response["llm"]["replies"][0]
      final_answer = _extract_reply_text(reply)

      contexts = [doc.content for doc in response["retriever"]["documents"]]
      usage = reply.meta.get("usage", {}) if hasattr(reply, "meta") else {}

      results.append({
        "id": item["id"],
        "question": q,
        "ground_truth": item["ground_truth"],
        "expected_source": item["expected_source"],
        "Configuration": config_label,
        "Chunker": chunker_name,
        "Embedder": embedder_model,
        "LLM": llm_name,
        "contexts": contexts,
        "answer": final_answer,
        "latency": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
      })
    except Exception as exc:
      print(f"  Error on '{config_label}' / question '{q}': {exc}")

  return results
