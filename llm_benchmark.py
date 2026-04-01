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

    # Panel 1: Avg Accuracy by Strategy
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Average Accuracy by Strategy")
    for i, row in summary.iterrows():
        s = row["strategy"]
        bar = ax1.bar(s, row["avg_accuracy"], color=color_map[s],
                      alpha=0.85, width=0.5, zorder=3)
        ax1.text(i - 1, row["avg_accuracy"] + 0.005,
                 f"{row['avg_accuracy']:.1%}",
                 ha="center", va="bottom", color=WHITE, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Accuracy", color=MUTED, fontsize=9)
    ax1.set_xticklabels(summary["strategy"], rotation=12, ha="right", color=MUTED)
    ax1.axhline(0.8, color="#f78166", linewidth=1, linestyle="--", alpha=0.6,
                label="0.80 threshold")
    ax1.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL, edgecolor=BORDER)

    # Panel 2: Accuracy vs Latency scatter
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Accuracy vs Latency (all tasks)")
    ax2.grid(axis="both", color=BORDER, linewidth=0.6, alpha=0.7)

    for strategy in strategies:
        sub = df[df["strategy"] == strategy]
        ax2.scatter(sub["latency_s"], sub["accuracy"],
                    color=color_map[strategy], alpha=0.7, s=55,
                    label=strategy, zorder=3, edgecolors="none")

    ax2.set_xlabel("Latency (s)", color=MUTED, fontsize=9)
    ax2.set_ylabel("Accuracy", color=MUTED, fontsize=9)
    ax2.legend(fontsize=8, labelcolor=WHITE, facecolor=PANEL,
               edgecolor=BORDER, loc="lower right")

    # Panel 3: Accuracy by Task Type heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL)
    ax3.set_title("Accuracy Heatmap: Strategy × Task Type",
                  color=WHITE, fontsize=12, fontweight="bold", pad=10)

    pivot = df.pivot_table(index="strategy", columns="task_type",
                           values="accuracy", aggfunc="mean")
    pivot = pivot.reindex(strategies)

    im = ax3.imshow(pivot.values, cmap="Blues", aspect="auto", vmin=0.4, vmax=1.0)

    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_xticklabels(pivot.columns, rotation=35, ha="right",
                        color=MUTED, fontsize=8)
    ax3.set_yticklabels(pivot.index, color=MUTED, fontsize=8)
    ax3.tick_params(colors=MUTED)
    for spine in ax3.spines.values():
        spine.set_color(BORDER)

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            ax3.text(c, r, f"{val:.0%}", ha="center", va="center",
                     color=WHITE if val > 0.65 else MUTED, fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)

    # Panel 4: Efficiency Score (accuracy, latency)
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Efficiency Score (Accuracy ÷ Latency)")

    bars = ax4.barh(summary["strategy"][::-1],
                    summary["efficiency_score"][::-1],
                    color=[color_map[s] for s in summary["strategy"][::-1]],
                    alpha=0.85, height=0.45, zorder=3)

    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f}", va="center", color=WHITE, fontsize=9, fontweight="bold")

    ax4.set_xlabel("Efficiency (higher = better value)", color=MUTED, fontsize=9)
    ax4.tick_params(axis="y", colors=MUTED)
    ax4.grid(axis="x", color=BORDER, linewidth=0.6, alpha=0.7)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n[✓] Dashboard saved to '{output_path}'")

# --- print a clean results table to terminal ---

def print_summary_table(summary: pd.DataFrame):
    print("\n" + "═" * 75)
    print("  BENCHMARK RESULTS SUMMARY")
    print("═" * 75)
    print(f"  {'Rank':<5} {'Strategy':<20} {'Accuracy':>10} {'Latency':>10} {'Efficiency':>12} {'Est. Cost':>12}")
    print("  " + "─" * 70)
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for rank, row in summary.iterrows():
        medal = medals.get(rank, "  ")
        print(
            f"  {medal} {rank:<3} {row['strategy']:<20} "
            f"{row['avg_accuracy']:>9.1%} "
            f"{row['avg_latency']:>9.2f}s "
            f"{row['efficiency_score']:>12.4f} "
            f"${row['total_cost']:>10.4f}"
        )
    print("═" * 75)
    top = summary.iloc[0]
    print(f"\n  Best overall: {top['strategy']} ({top['avg_accuracy']:.1%} accuracy)")
    print(f"  Most efficient: {summary.sort_values('efficiency_score', ascending=False).iloc[0]['strategy']}")
    print()

# --- run it ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Prompt Strategy Benchmarker")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-task output")
    parser.add_argument("--output", type=str, default="benchmark_results.png",
                        help="Output path for dashboard image")
    args = parser.parse_args()

    df = run_benchmark(verbose=not args.quiet)
    summary = analyze(df)
    print_summary_table(summary)

    df.to_csv("benchmark_raw.csv", index=False)
    print(f"[✓] Raw results saved to 'benchmark_raw.csv'")

    visualize(df, summary, output_path=args.output)
    print(f"\nDone. Open '{args.output}' to view the dashboard.\n")
