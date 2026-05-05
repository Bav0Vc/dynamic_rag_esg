from hypster import HP


def chunking_config(hp: HP):
  chunker_name = hp.select(
    [
      "RecursiveSplitter",
      "FixedSizeWordSplitter",
      "SemanticEmbeddingChunker"
    ],
    name="chunker_name",
    default="RecursiveSplitter",
  )
  return { "chunker_name": chunker_name }

def embedding_config(hp: HP):
  model = hp.select(
    [
      "BAAI/bge-m3",
      "Snowflake/snowflake-arctic-embed-l-v2.0",
      "intfloat/multilingual-e5-large-instruct",
    ],
    name="model",
    default="BAAI/bge-m3",
  )

  _prefixes = {
    "intfloat/multilingual-e5-large-instruct": ("query: ", "passage: "),
  }
  query_prefix, doc_prefix = _prefixes.get(model, ("", ""))

  _dims = {
    "BAAI/bge-m3": 1024,
    "Snowflake/snowflake-arctic-embed-l-v2.0": 256,  # MRL truncation
    "intfloat/multilingual-e5-large-instruct": 1024,
  }
  # Only set for models that use MRL truncation below their native dimension
  _truncate_dim = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": 256,
  }

  return {
    "model": model,
    "api_model": model,
    "backend": "sentence-transformers",
    "dims": _dims[model],
    "truncate_dim": _truncate_dim.get(model),
    "query_prefix": query_prefix,
    "doc_prefix": doc_prefix,
  }

def llm_config(hp: HP):
  configs = {
    "Qwen-2.5-14B-Instruct": {
      "backend": "hf",
      "api_model": "Qwen/Qwen2.5-14B-Instruct"
    },
    "Llama-3.3-70B-Instruct": {
      "backend": "hf",
      "api_model": "meta-llama/Llama-3.3-70B-Instruct"
    },
    "Mistral-Large-2": {
      "backend": "mistral",
      "api_model": "mistral-large-latest"
    },
  }

  name = hp.select(list(configs.keys()), name="name", default="Qwen-2.5-14B-Instruct")
    
  # Merge the name key with the selected configuration dictionary
  return { "name": name, **configs[name] }

def pipeline_config(hp: HP):
  chunking = hp.nest(chunking_config, name="chunking")
  embedding = hp.nest(embedding_config, name="embedding")
  llm = hp.nest(llm_config, name="llm")
  return hp.collect(locals())