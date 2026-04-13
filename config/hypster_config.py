from hypster import HP, instantiate


def chunking_config(hp: HP):
    chunker_name = hp.select(
        ["RecursiveSplitter", "FixedSizeWordSplitter", "SemanticEmbeddingChunker"],
        name="chunker_name",
        default="RecursiveSplitter",
    )
    return hp.collect(locals())


def embedding_config(hp: HP):
    model = hp.select(
        [
            "BAAI/bge-m3",
            "text-embedding-3-large",
            "intfloat/multilingual-e5-large-instruct",
        ],
        name="model",
        default="BAAI/bge-m3",
    )
    return hp.collect(locals())


def llm_config(hp: HP):
    name = hp.select(
        ["GPT-4o-mini", "Gemini-2.5-Flash", "Mistral-Large-2"],
        name="name",
        default="GPT-4o-mini",
    )
    return hp.collect(locals())


def pipeline_config(hp: HP):
    chunking = hp.nest(chunking_config, name="chunking")
    embedding = hp.nest(embedding_config, name="embedding")
    llm = hp.nest(llm_config, name="llm")
    return hp.collect(locals())
