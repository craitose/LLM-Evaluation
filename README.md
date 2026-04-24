This project provides a robust pipeline for evaluating Large Language Model (LLM) outputs using the Ragas framework.

It supports both OpenAI (for high-accuracy judging) and Ollama (for 100% local, private evaluation).

## Features

Dual Mode: Switch between OpenAI (GPT-4o) and Local (Qwen 2.5) with a single toggle.

RAG Metrics: Evaluates Faithfulness (hallucination check) and Answer Correctness.

Local Priority: Optimized for qwen2.5:3b via Ollama to run on standard hardware.

Async Implementation: Uses asynchronous clients for better performance.

## Prerequisites

Python 3.10 - 3.13 (Note: Python 3.14+ currently has Pydantic compatibility issues).

Ollama (for local evaluation).

OpenAI API Key (for cloud evaluation).

## Installation
1.Clone the repository:

git clone https://github.com/craitose/LLM-Evaluation.git

cd LLM-EVALUATION

2.Install dependencies:

pip install -r requirements.txt

3.Set up environment variables:

Create a .env file in the root directory:

OPENAI_API_KEY=sk-your-actual-key-here

4.Pull the local model (if using Local Mode):

ollama pull qwen2.5:3b


## Usage

1.Configure Mode: Open app.py and set MODE = 'local' or MODE = 'openai'.

2.Run the script:

python app.py

3.View Results: Scores will be printed to the console and saved to score.csv.

## Evaluation Metrics

Faithfulness: Measures if the answer is factually supported by the retrieved context.
Answer Correctness: Measures the semantic similarity between the generated answer and the ground truth.

Author: @craitose
