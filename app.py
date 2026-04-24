from datasets import Dataset

from openai import AsyncOpenAI
from ragas import evaluate, RunConfig
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics import _faithfulness, _answer_correctness

run_config = RunConfig(max_workers=1)

client = AsyncOpenAI(api_key=("sk-XXXXXXXXXXXXXXXX"))

# Ragas uses these under the hood, but explicit wrapping solves the type mismatch
ragas_llm = llm_factory(model="gpt-4o", client=client)
ragas_emb = embedding_factory("openai", model="text-embedding-3-small", client=client)


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
