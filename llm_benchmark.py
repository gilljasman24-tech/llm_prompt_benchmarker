# llm_benchmark.py
# I built this to compare different prompt strategies on business tasks
# and figure out which ones actually perform better and why.
# no API key needed, I simulated the responses instead
# Author: Jasman Gill

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import random
import argparse
from datetime import datetime

# --- tasks and strategies I'm testing ---

TASKS = [
    {"id": "T01", "type": "Summarization",     "difficulty": "easy",   "input": "Summarize quarterly earnings report"},
    {"id": "T02", "type": "Classification",    "difficulty": "medium", "input": "Classify customer complaint severity"},
    {"id": "T03", "type": "Data Extraction",   "difficulty": "hard",   "input": "Extract structured fields from unstructured invoice"},
    {"id": "T04", "type": "Sentiment Analysis","difficulty": "easy",   "input": "Detect sentiment of product review"},
    {"id": "T05", "type": "Code Generation",   "difficulty": "hard",   "input": "Write SQL query from natural language"},
    {"id": "T06", "type": "Summarization",     "difficulty": "medium", "input": "Summarize 50-page policy document"},
    {"id": "T07", "type": "Classification",    "difficulty": "easy",   "input": "Tag support tickets by department"},
    {"id": "T08", "type": "Data Extraction",   "difficulty": "medium", "input": "Pull dates and amounts from contract PDF"},
    {"id": "T09", "type": "Reasoning",         "difficulty": "hard",   "input": "Identify root cause from error logs"},
    {"id": "T10", "type": "Reasoning",         "difficulty": "medium", "input": "Suggest process improvement from KPI data"},
]

STRATEGIES = {
    "Zero-Shot": {
        "description": "Direct prompt with no examples or guidance.",
        "base_accuracy": 0.61,
        "base_latency": 0.9,
    },
    "Few-Shot": {
        "description": "Prompt includes 3 worked examples before the task.",
        "base_accuracy": 0.74,
        "base_latency": 1.4,
    },
    "Chain-of-Thought": {
        "description": "Prompt instructs the model to reason step-by-step.",
        "base_accuracy": 0.81,
        "base_latency": 2.1,
    },
    "Role + CoT": {
        "description": "Assigns a domain expert role, then reasons step-by-step.",
        "base_accuracy": 0.87,
        "base_latency": 2.5,
    },
}

# harder tasks drag accuracy down, easier ones boost it
DIFFICULTY_MODIFIER = {"easy": +0.08, "medium": 0.0, "hard": -0.12}

# --- simulation logic ---

def simulate_response(strategy_name: str, task: dict, seed: int) -> dict:
    # fake a model response for each strategy/task combo
    # using a fixed seed so results are reproducible
    rng = np.random.default_rng(seed)
    strategy = STRATEGIES[strategy_name]

    accuracy = (
        strategy["base_accuracy"]
        + DIFFICULTY_MODIFIER[task["difficulty"]]
        + rng.normal(0, 0.06)
    )
    accuracy = float(np.clip(accuracy, 0.0, 1.0))

    latency = strategy["base_latency"] + rng.normal(0, 0.3)
    latency = float(max(0.2, latency))

    tokens = int(rng.integers(80, 600))
    cost_per_1k = 0.003  # rough estimate based on Claude Haiku pricing
    cost = round((tokens / 1000) * cost_per_1k, 6)

    return {
        "strategy": strategy_name,
        "task_id": task["id"],
        "task_type": task["type"],
        "difficulty": task["difficulty"],
        "accuracy": round(accuracy, 4),
        "latency_s": round(latency, 3),
        "tokens_used": tokens,
        "estimated_cost_usd": cost,
    }


def run_benchmark(verbose: bool = True) -> pd.DataFrame:
    # loop every strategy over every task and collect results
    results = []
    total = len(STRATEGIES) * len(TASKS)
    done = 0

    print("\n" + "═" * 55)
    print("  LLM PROMPT STRATEGY BENCHMARKER")
    print("═" * 55)
    print(f"  Strategies : {len(STRATEGIES)}")
    print(f"  Tasks      : {len(TASKS)}")
    print(f"  Total runs : {total}")
    print("═" * 55)

    for strategy_name in STRATEGIES:
        if verbose:
            print(f"\n▶ Running: {strategy_name}")
            print(f"  {STRATEGIES[strategy_name]['description']}")

        for i, task in enumerate(TASKS):
            seed = hash(f"{strategy_name}{task['id']}") % (2**31)
            result = simulate_response(strategy_name, task, seed)
            results.append(result)
            done += 1

            if verbose:
                bar = "█" * int(result["accuracy"] * 20) + "░" * (20 - int(result["accuracy"] * 20))
                print(f"  [{task['id']}] {task['type']:<20} acc={result['accuracy']:.2%}  {bar}")

            time.sleep(0.03)  # small delay so it feels like real API calls

    df = pd.DataFrame(results)
    return df