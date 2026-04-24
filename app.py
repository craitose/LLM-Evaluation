import os
from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import evaluate, RunConfig
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics import _faithfulness, _answer_correctness

load_dotenv()
run_config = RunConfig(max_workers=1, timeout=60)

# TOGGLE THIS: 'openai' or 'local'
MODE = "local"

if MODE == "openai":
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ragas_llm = llm_factory(model="gpt-4o", client=client)
    ragas_emb = embedding_factory(
        "openai", model="text-embedding-3-small", client=client
    )
else:
    # For Local Llama via Ollama

    local_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama doesn't need a real key, but the client requires a string
    )
    local_model = (
        "qwen2.5:3b"  # Ensure this matches the model name in your Ollama setup
    )
    ragas_llm = llm_factory(model=local_model, client=local_client)
    ragas_emb = embedding_factory(
        provider="openai", model=local_model, client=local_client
    )


data_samples = {
    "question": [
        "What is the capital of France?",
        "Who is the president of the United States?",
        "What is the largest mammal?",
    ],
    "answer": [
        "The capital of France is Paris.",
        "The president of the United States is Joe Biden.",
        "The largest mammal is the blue whale.",
    ],
    "contexts": [
        [
            "Paris is the capital city of France, known for its art, culture, and history."
        ],
        [
            "Joe Biden is the 46th president of the United States, serving since January 20, 2021."
        ],
        [
            "The blue whale is the largest mammal on Earth, reaching lengths of up to 100 feet and weights of up to 200 tons."
        ],
    ],
    "ground_truth": ["Paris", "Joe Biden", "Blue whale"],
}

dataset = Dataset.from_dict(data_samples)

score = evaluate(
    dataset,
    metrics=[_faithfulness, _answer_correctness],
    llm=ragas_llm,
    embeddings=ragas_emb,
    run_config=run_config,
)
df = score.to_pandas()
print(df)
df.to_csv("score.csv", index=False)
