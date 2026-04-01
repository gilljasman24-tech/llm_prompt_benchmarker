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

# --- aggregate the results per strategy ---

def analyze(df: pd.DataFrame) -> pd.DataFrame:
    # roll up per-task results into a single summary row per strategy
    summary = df.groupby("strategy").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_latency=("latency_s", "mean"),
        total_tokens=("tokens_used", "sum"),
        total_cost=("estimated_cost_usd", "sum"),
        tasks_run=("task_id", "count"),
    ).round(4).reset_index()

    # Efficiency score: accuracy per second of latency
    summary["efficiency_score"] = (
        summary["avg_accuracy"] / summary["avg_latency"]
    ).round(4)

    summary.sort_values("avg_accuracy", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    summary.index += 1  # rank starts at 1

    return summary

# --- charts ---

def visualize(df: pd.DataFrame, summary: pd.DataFrame, output_path: str = "benchmark_results.png"):
    # 4 panel dashboard: accuracy bar, scatter, heatmap, efficiency

    BG      = "#0d1117"
    PANEL   = "#161b22"
    BORDER  = "#30363d"
    WHITE   = "#e6edf3"
    MUTED   = "#8b949e"
    COLORS  = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]

    strategies = list(STRATEGIES.keys())
    color_map = {s: COLORS[i] for i, s in enumerate(strategies)}

    fig = plt.figure(figsize=(18, 11), facecolor=BG)
    fig.suptitle(
        "LLM Prompt Strategy Benchmark  ·  Jasman Gill",
        fontsize=17, fontweight="bold", color=WHITE, y=0.97
    )
    subtitle = f"4 strategies  ·  {len(TASKS)} tasks  ·  {datetime.now().strftime('%Y-%m-%d')}"
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10, color=MUTED)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.32,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=10)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(axis="y", color=BORDER, linewidth=0.6, alpha=0.7)