import os
import sys
import time

from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import OpenAITextEmbedder, HuggingFaceAPITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
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

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_HF_LLM_BASE_URL = "https://router.huggingface.co/featherless-ai/v1"


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


def _build_text_embedder(emb_cfg: dict):
  if emb_cfg["backend"] == "nvidia":
    return OpenAITextEmbedder(
      model=emb_cfg["api_model"],
      api_key=Secret.from_env_var("NVIDIA_API_KEY"),
      api_base_url=_NVIDIA_BASE_URL,
    )
  return HuggingFaceAPITextEmbedder(
    api_type="serverless_inference_api",
    api_params={"model": emb_cfg["api_model"]},
    token=Secret.from_env_var("HF_TOKEN"),
  )


def _build_llm(llm_cfg: dict):
  if llm_cfg["backend"] == "mistral":
    return MistralChatGenerator(model=llm_cfg["api_model"])
  if llm_cfg["backend"] == "nvidia":
    return OpenAIGenerator(
      model=llm_cfg["api_model"],
      api_key=Secret.from_env_var("NVIDIA_API_KEY"),
      api_base_url=_NVIDIA_BASE_URL,
    )
  return OpenAIGenerator(
    model=llm_cfg["api_model"],
    api_key=Secret.from_env_var("HF_TOKEN"),
    api_base_url=_HF_LLM_BASE_URL,
  )


def run_query_pipeline(config: dict, golden_dataset: list) -> list:
  chunker_name: str = config["chunking"]["chunker_name"]
  emb_cfg: dict = config["embedding"]
  llm_cfg: dict = config["llm"]

  embedder_model = emb_cfg["model"]
  llm_name = llm_cfg["name"]
  config_label = f"{chunker_name} | {embedder_model} | {llm_name}"
  print(f"Running config: {config_label}")

  collection_name = f"{chunker_name}_{embedder_model}".replace("/", "-").lower()

  document_store = QdrantDocumentStore(
    url=os.getenv("QDRANT_URL"),
    api_key=Secret.from_env_var("QDRANT_API_KEY"),
    index=collection_name,
    embedding_dim=emb_cfg["dims"],
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

  text_embedder = _build_text_embedder(emb_cfg)
  llm_instance = _build_llm(llm_cfg)

  query_pipe = Pipeline()
  query_pipe.add_component("text_embedder", text_embedder)
  query_pipe.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store))
  query_pipe.add_component("prompt_builder", PromptBuilder(template=_PROMPT_TEMPLATE, required_variables=["documents", "question"]))
  query_pipe.add_component("llm", llm_instance)

  query_pipe.connect("text_embedder.embedding", "retriever.query_embedding")
  query_pipe.connect("retriever.documents", "prompt_builder.documents")
  query_pipe.connect("prompt_builder", "llm")

  results = []
  for item in golden_dataset:
    q = item["answer"]["user_question"]
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
      # MistralChatGenerator returns ChatMessage with .meta; OpenAIGenerator returns a
      # plain string and puts usage in response["llm"]["meta"][0]
      if hasattr(reply, "meta"):
        usage = reply.meta.get("usage", {})
      else:
        meta_list = response["llm"].get("meta", [{}])
        usage = (meta_list[0].get("usage", {}) if meta_list else {})

      results.append({
        "question_id": item["question_id"],
        "question": q,
        "requirement": item["question"],
        "ground_truth": item["ground_truth"],
        "expected_source": item["expected_source"],
        "reference_contexts": list(item.get("reference_contexts", {}).values()),
        "source_page": item.get("source_page"),
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
