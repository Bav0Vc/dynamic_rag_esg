from hypster import HP, instantiate


def chunking_config(hp: HP):
  chunker_name = hp.select(
    ["RecursiveSplitter", "FixedSizeWordSplitter", "SemanticEmbeddingChunker"],
    name="chunker_name",
    default="RecursiveSplitter",
  )
  return {"chunker_name": chunker_name}


def embedding_config(hp: HP):
  model = hp.select(
    [
      "BAAI/bge-m3",
      "snowflake/arctic-embed-1-v2.0",
      "intfloat/multilingual-e5-large-instruct",
    ],
    name="model",
    default="BAAI/bge-m3",
  )
  if model == "BAAI/bge-m3":
    backend = "nvidia"
    api_model = "baai/bge-m3"
  elif model == "snowflake/arctic-embed-1-v2.0":
    backend = "hf"
    api_model = "Snowflake/snowflake-arctic-embed-l-v2.0"
  else:
    backend = "hf"
    api_model = "intfloat/multilingual-e5-large"
  return {"model": model, "backend": backend, "api_model": api_model, "dims": 1024}


def llm_config(hp: HP):
  name = hp.select(
    [
      "Qwen-2.5-14B",
      "Llama-3.3-70B",
      "Mistral-Large-2",
    ],
    name="name",
    default="Qwen-2.5-14B",
  )
  if name == "Qwen-2.5-14B":
    backend = "hf"
    api_model = "Qwen/Qwen2.5-14B-Instruct"
  elif name == "Llama-3.3-70B":
    backend = "nvidia"
    api_model = "meta/llama-3.3-70b-instruct"
  else:  # Mistral-Large-2
    backend = "mistral"
    api_model = "mistral-large-latest"
  return {"name": name, "backend": backend, "api_model": api_model}


def pipeline_config(hp: HP):
  chunking = hp.nest(chunking_config, name="chunking")
  embedding = hp.nest(embedding_config, name="embedding")
  llm = hp.nest(llm_config, name="llm")
  return hp.collect(locals())
