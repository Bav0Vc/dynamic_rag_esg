import os
import json
import pandas as pd
import numpy as np
from ragas import evaluate
from dotenv import load_dotenv

# from ragas.metrics import (
#   Faithfulness,
#   AnswerRelevancy,
#   ContextPrecision,
#   ContextRecall
# )
from ragas.metrics.collections import Faithfulness
from ragas.metrics.collections import AnswerRelevancy
from ragas.metrics.collections import ContextPrecision
from ragas.metrics.collections import ContextRecall

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from langchain_openai import ChatOpenAI
from langchain_mistralai import MistralAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

def calculate_cost(llm_name, prompt_tokens, completion_tokens):
  # Prices per 1M tokens
  prices = {
    "GPT-4o-mini": {"in": 0.150, "out": 0.600},
    "Gemini-2.5-Flash": {"in": 0.075, "out": 0.300},
    "Mistral-Large-2": {"in": 2.000, "out": 6.000}
  }
  
  p = prices.get(llm_name, {"in": 0, "out": 0})
  return (prompt_tokens * p["in"] + completion_tokens * p["out"]) / 1_000_000

def evaluate_results():
  input_file = "evaluation/results/evaluation_dataset.json"
  if not os.path.exists(input_file):
    print(f"Data not found at {input_file}. Run query_pipeline.py first.")
    return

  with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

  df = pd.DataFrame(data)
  
  # Process custom metrics for all rows
  df["cost"] = df.apply(lambda row: calculate_cost(row["LLM"], row["prompt_tokens"], row["completion_tokens"]), axis=1)
  df["source_attribution"] = df.apply(lambda row: 1.0 if row["expected_source"] in row["answer"] else 0.0, axis=1)
  
  leaderboard = []

  # Process per configuration
  configurations = df["Configuration"].unique()
  
  # Initialize evaluators
  mistral_api_key = os.getenv("MISTRAL_API_KEY")
  evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    api_key=mistral_api_key, 
    base_url="https://api.mistral.ai/v1", 
    model="mistral-large-2407"
  ))
  evaluator_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(mistral_api_key=mistral_api_key, model="mistral-embed"))
  
  for config in configurations:
    print(f"Evaluating config: {config}")
    subset = df[df["Configuration"] == config]
    
    # Construct Ragas format items
    samples = []
    for _, row in subset.iterrows():
      sample = SingleTurnSample(
        user_input=row["question"],
        response=row["answer"],
        retrieved_contexts=row["contexts"],
        reference=row["ground_truth"]
      )
      samples.append(sample)

    dataset = EvaluationDataset(samples=samples)
    
    try:
      # Using Mistral as evaluator judge
      metrics = [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]
      
      ragas_result = evaluate(
        dataset, 
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
      )
      
      scores_df = ragas_result.to_pandas()
      
      leaderboard.append({
        "Configuration": config,
        "Chunker": subset.iloc[0]["Chunker"],
        "Embedder": subset.iloc[0]["Embedder"],
        "LLM": subset.iloc[0]["LLM"],
        "Latency (s)": round(subset["latency"].mean(), 3),
        "Cost per Query ($)": round(subset["cost"].mean(), 6),
        "Source Attribution": round(subset["source_attribution"].mean(), 2),
        "Faithfulness": round(scores_df["faithfulness"].mean(), 2) if "faithfulness" in scores_df else np.nan,
        "Answer Relevancy": round(scores_df["answer_relevancy"].mean(), 2) if "answer_relevancy" in scores_df else np.nan,
        "Context Recall": round(scores_df["context_recall"].mean(), 2) if "context_recall" in scores_df else np.nan,
        "Context Precision": round(scores_df["context_precision"].mean(), 2) if "context_precision" in scores_df else np.nan
      })
    except Exception as e:
      print(f"Failed to evaluate {config}: {e}")

  df_leaderboard = pd.DataFrame(leaderboard)
  os.makedirs("evaluation/results", exist_ok=True)
  df_leaderboard.to_csv("evaluation/results/metrics_leaderboard.csv", index=False, sep=";")
  print("Leaderboard generation complete. Saved to evaluation/results/metrics_leaderboard.csv")
  
  print("\n--- Leaderboard Summary ---")
  print(df_leaderboard)

if __name__ == "__main__":
  evaluate_results()