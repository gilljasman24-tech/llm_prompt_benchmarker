# llm_benchmark.py
# I built this to compare different prompt strategies on business tasks
# and figure out which ones actually perform better and why.
# every run calls a real model (Claude) through the Anthropic API and checks
# the answer against a known right answer, so the scores are measured, not made up.
# needs ANTHROPIC_API_KEY set. a full run is 40 calls and costs well under $1.
# Author: Jasman Gill

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import re
import sqlite3
import time
import argparse
from datetime import datetime

import anthropic

# --- model settings ---

MODEL = "claude-opus-5-5"
EFFORT = "low"                   # same for every strategy so only the prompt changes
PRICE_IN_PER_M = 4.00            # $ per 1M input tokens for this model
PRICE_OUT_PER_M = 20.00          # $ per 1M output tokens (thinking counts as output)

# --- tasks: each one has a real input, a known answer, and a way to score it ---
# score functions return 0.0 to 1.0

def has_all(*needles):
    # summary tasks: share of the key facts that made it into the answer.
    # a needle can be a tuple of accepted spellings, like ("january 1", "jan 1")
    def score(answer):
        a = answer.lower().replace(",", "")
        options = [n if isinstance(n, tuple) else (n,) for n in needles]
        return sum(1 for opts in options if any(o.lower() in a for o in opts)) / len(options)
    return score

def one_of(*labels):
    # label tasks: exact label match, ignoring case and punctuation
    def score(answer):
        a = re.sub(r"[^a-z ]", "", answer.lower()).strip()
        return 1.0 if a in labels else 0.0
    return score

def contains_any(*needles):
    def score(answer):
        a = answer.lower()
        return 1.0 if any(n.lower() in a for n in needles) else 0.0
    return score

def invoice_fields(answer):
    expected = {
        "invoice_number": "INV-20931",
        "vendor": "harbour lane printing co",
        "issue_date": "2026-09-03",
        "due_date": "2026-10-03",
        "total": 609.00,
    }
    m = re.search(r"\{.*\}", answer, re.S)
    if not m:
        return 0.0
    try:
        got = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.0
    right = 0
    for key, want in expected.items():
        val = got.get(key)
        if key == "total":
            try:
                right += abs(float(str(val).replace("$", "").replace(",", "")) - want) < 0.01
            except (TypeError, ValueError):
                pass
        elif key == "vendor":
            right += str(val).lower().rstrip(".") == want
        else:
            right += str(val).strip() == want
    return right / len(expected)

def contract_items(answer):
    # every date and dollar amount in the lease, no extras
    expected = {"2026-03-01", "2027-02-28", "2026-02-15", "2150", "1075", "75"}
    found = set(re.findall(r"\d{4}-\d{2}-\d{2}", answer))
    found |= {n.replace(",", "") for n in re.findall(r"\$\s?([\d,]+)(?:\.00)?", answer)}
    hits = len(found & expected)
    return hits / max(len(expected), len(found))

# a tiny database for the SQL task, so the model's query can actually be run
SQL_SETUP = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, total REAL, status TEXT);
INSERT INTO customers VALUES (1,'Avery Shah','Calgary'),(2,'Ben Ortiz','Calgary'),(3,'Chloe Ng','Edmonton'),(4,'Dev Patel','Calgary');
INSERT INTO orders VALUES
 (1,1,'2026-01-14',300,'completed'),(2,1,'2026-02-02',250,'completed'),(3,2,'2026-03-09',900,'cancelled'),
 (4,2,'2026-04-21',400,'completed'),(5,3,'2026-05-30',2000,'completed'),(6,4,'2025-12-28',1200,'completed'),
 (7,4,'2026-06-11',350,'completed'),(8,1,'2026-07-04',100,'refunded'),(9,2,'2026-08-19',180,'completed');
"""
SQL_SCHEMA = ("customers(id, name, city)\n"
              "orders(id, customer_id, order_date TEXT 'YYYY-MM-DD', total REAL, status TEXT: 'completed' | 'cancelled' | 'refunded')")

def sql_result(answer):
    # right answer is Ben Ortiz ($580 completed in 2026). Avery has $550; Dev's big order was in 2025;
    # Ben's $900 was cancelled. the query has to get all three of those right.
    m = re.search(r"(select .*?)(?:```|$)", answer, re.I | re.S)
    if not m:
        return 0.0
    db = sqlite3.connect(":memory:")
    db.executescript(SQL_SETUP)
    try:
        rows = db.execute(m.group(1).strip().rstrip(";")).fetchall()
    except sqlite3.Error:
        return 0.0
    return 1.0 if rows and rows[0][0] == "Ben Ortiz" else 0.0

TASKS = [
    {"id": "T01", "type": "Summarization", "difficulty": "easy", "role": "senior financial analyst",
     "instruction": "Summarize this earnings update in 2 sentences for a busy executive. Keep every key number.",
     "input": "Q3 update: Northwind Supplies posted revenue of $4.2 million in Q3, up 12% from Q2, driven mainly "
              "by its new online store. Operating costs rose 5% because of higher shipping rates. Net profit was "
              "$610,000. The company expects Q4 revenue between $4.5 and $4.8 million and plans to hire 6 warehouse staff.",
     "examples": [
         ("Q1 update: Lakeside Bakery made $820,000 in Q1, down 3% from Q4 after a slow January. Flour costs fell 8%. "
          "Profit was $95,000. It expects Q2 sales of about $900,000.",
          "Lakeside Bakery made $820,000 in Q1 (down 3%) with $95,000 profit, helped by 8% lower flour costs. It expects about $900,000 in Q2."),
         ("Q2 update: Pine Ridge Tours earned $1.1 million in Q2, up 40% with the summer season. Fuel costs rose 15%. "
          "Profit was $210,000 and it will add 2 buses.",
          "Pine Ridge Tours earned $1.1 million in Q2 (up 40%) and $210,000 profit despite 15% higher fuel costs. It plans to add 2 buses."),
     ],
     "score": has_all("4.2", "12%", "610", "4.5", "4.8")},

    {"id": "T02", "type": "Classification", "difficulty": "medium", "role": "customer support lead",
     "instruction": "Classify the severity of this complaint. Rules: critical = safety risk or an outage hitting many "
                    "customers; high = one customer losing money or threatening a chargeback or legal action; "
                    "medium = something broken but there is a workaround; low = a question or cosmetic issue. "
                    "Answer with one word: low, medium, high, or critical.",
     "input": "I've been charged twice for my annual plan ($480 each). I contacted support 9 days ago and nobody "
              "replied. If this isn't fixed by Friday I'm disputing it with my bank.",
     "examples": [
         ("The export button is greyed out but I can still copy the table by hand. Annoying.", "medium"),
         ("Your whole site has been down for an hour and none of our 200 staff can log in.", "critical"),
     ],
     "score": one_of("high")},

    {"id": "T03", "type": "Data Extraction", "difficulty": "hard", "role": "accounts payable specialist",
     "instruction": "Extract these fields as JSON: invoice_number, vendor, issue_date, due_date, total. "
                    "Dates in the invoice are written DD/MM/YYYY. Give dates as YYYY-MM-DD and total as a number.",
     "input": "INVOICE   #INV-20931\nBill from: Harbour Lane Printing Co.\nIssued 03/09/2026 (payment due in 30 days)\n"
              "500 x Flyers A5 ............ $210.00\n2 x Banner 6ft ............. $185.00 each\n"
              "Subtotal $580.00\nGST 5% $29.00\nTOTAL DUE $609.00",
     "examples": [
         ("INVOICE #A-77\nFrom: Bow Valley Signs\nDate 15/01/2026, due in 14 days\nTotal: $1,050.00",
          '{"invoice_number": "A-77", "vendor": "Bow Valley Signs", "issue_date": "2026-01-15", "due_date": "2026-01-29", "total": 1050.00}'),
         ("Inv no. 5521 | Kestrel Office Supply | 02/02/2026 | net 30 | amount due $88.40",
          '{"invoice_number": "5521", "vendor": "Kestrel Office Supply", "issue_date": "2026-02-02", "due_date": "2026-03-04", "total": 88.40}'),
     ],
     "score": invoice_fields},

    {"id": "T04", "type": "Sentiment Analysis", "difficulty": "easy", "role": "customer insights analyst",
     "instruction": "What is the overall sentiment of this review? Answer with one word: positive, negative, or neutral.",
     "input": "Arrived two weeks late, box crushed, and the 'waterproof' jacket soaked through on the first rainy "
              "walk. At least the colour is nice.",
     "examples": [
         ("Fast shipping and it fits perfectly. Wearing it every day.", "positive"),
         ("It's a phone case. It holds the phone. Nothing special either way.", "neutral"),
     ],
     "score": one_of("negative")},

    {"id": "T05", "type": "Code Generation", "difficulty": "hard", "role": "senior data engineer",
     "instruction": "Write one SQLite query for this question. Tables:\n" + SQL_SCHEMA +
                    "\nQuestion: which customer in Calgary spent the most on completed orders placed in 2026? "
                    "Return just their name. Give only the SQL as your answer.",
     "input": "(the database above)",
     "examples": [
         ("Question: how many orders were cancelled?", "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'"),
         ("Question: total completed sales in Edmonton?",
          "SELECT SUM(o.total) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.city = 'Edmonton' AND o.status = 'completed'"),
     ],
     "score": sql_result},

    {"id": "T06", "type": "Summarization", "difficulty": "medium", "role": "HR policy advisor",
     "instruction": "Summarize this policy in 3 short bullet points an employee can skim. Keep the specific numbers and dates.",
     "input": "Hybrid Work Policy (effective January 1, 2027). Employees in eligible roles may work remotely up to "
              "2 days per week and must be in the office the other 3 days, including every Tuesday for team "
              "meetings. Fully remote arrangements need written manager approval and are reviewed every 6 months. "
              "The company provides a one-time $500 home office stipend, claimed through the expense system within "
              "90 days of starting hybrid work. Equipment bought with the stipend belongs to the employee.",
     "examples": [
         ("Vacation Policy (from March 1). Staff get 15 days a year, plus 1 extra day per year of service up to 20. "
          "Up to 5 unused days carry over.",
          "- From March 1: 15 vacation days a year, +1 per year of service, max 20\n- Up to 5 unused days carry over"),
         ("Parking Policy. Permits cost $40 a month, paid by payroll. Visitor spots are free for 2 hours.",
          "- Parking permits are $40 a month, taken from your pay\n- Visitor spots are free for 2 hours"),
     ],
     "score": has_all(("january 1", "jan 1"), ("3 days", "three days", "other 3"), "500", "90 days", "approval")},

    {"id": "T07", "type": "Classification", "difficulty": "easy", "role": "help desk dispatcher",
     "instruction": "Which department should handle this ticket? Answer with one word: billing, technical, sales, or hr.",
     "input": "Hi, the invoice you sent last month lists 12 seats but we only have 9 users. Can you correct it and "
              "refund the difference?",
     "examples": [
         ("The app crashes every time I upload a photo bigger than 5 MB.", "technical"),
         ("We're thinking about adding 50 more seats. Is there a volume discount?", "sales"),
     ],
     "score": one_of("billing")},

    {"id": "T08", "type": "Data Extraction", "difficulty": "medium", "role": "paralegal",
     "instruction": "List every date (as YYYY-MM-DD) and every dollar amount (as $ with no cents) in this contract, one per line.",
     "input": "This lease starts on March 1, 2026 and ends on February 28, 2027. Monthly rent is $2,150, due on the "
              "first of each month. A security deposit of $1,075 is due on signing, February 15, 2026. A late fee of "
              "$75 applies after the fifth day of the month.",
     "examples": [
         ("The loan of $5,000 is paid out on June 3, 2026 and repaid by June 3, 2028.", "2026-06-03\n2028-06-03\n$5000"),
         ("Membership is $120 a year, renewing every April 30. Signed April 30, 2026.", "2026-04-30\n$120"),
     ],
     "score": contract_items},

    {"id": "T09", "type": "Reasoning", "difficulty": "hard", "role": "site reliability engineer",
     "instruction": "What is the root cause of the 503 errors in these logs? Name the job or component responsible in a short phrase.",
     "input": "09:14:02 INFO  api    request /checkout 200 120ms\n"
              "09:14:05 WARN  db     pool: 48/50 connections in use\n"
              "09:14:07 INFO  worker report job started (export_all_orders)\n"
              "09:14:09 WARN  db     pool: 50/50 connections in use\n"
              "09:14:10 ERROR api    request /checkout 503 timeout waiting for db connection\n"
              "09:14:11 ERROR api    request /login 503 timeout waiting for db connection\n"
              "09:14:12 WARN  worker report job holding 30 connections (export_all_orders)\n"
              "09:14:15 ERROR api    request /checkout 503 timeout waiting for db connection\n"
              "09:15:40 INFO  worker report job finished (export_all_orders)\n"
              "09:15:41 INFO  db     pool: 12/50 connections in use\n"
              "09:15:42 INFO  api    request /checkout 200 98ms",
     "examples": [
         ("10:01 WARN disk 97% full on /var\n10:02 ERROR uploads failed: no space left\n10:05 INFO log rotation freed 40GB\n10:06 INFO uploads ok",
          "the disk filling up with logs (log rotation not running)"),
         ("08:00 INFO deploy v2.3 started\n08:01 ERROR /search 500 KeyError 'region'\n08:09 INFO rollback to v2.2\n08:10 INFO /search 200",
          "the v2.3 deploy (missing 'region' key)"),
     ],
     "score": contains_any("export_all_orders", "report job")},

    {"id": "T10", "type": "Reasoning", "difficulty": "medium", "role": "e-commerce analyst",
     "instruction": "Which region had the biggest drop in conversion rate (orders divided by visits) from August to "
                    "September? Answer with just the region name.",
     "input": "Region | Aug visits | Aug orders | Sep visits | Sep orders\n"
              "North  | 10000 | 400 | 12000 | 420\n"
              "South  |  8000 | 320 |  5000 | 210\n"
              "West   |  6000 | 300 |  6500 | 260\n"
              "East   |  9000 | 360 |  9000 | 351",
     "examples": [
         ("Store | Jan visits | Jan sales | Feb visits | Feb sales\nA | 100 | 10 | 100 | 5\nB | 200 | 10 | 100 | 6",
          "A"),
         ("Team | Q1 calls | Q1 deals | Q2 calls | Q2 deals\nRed | 50 | 5 | 40 | 6\nBlue | 80 | 8 | 80 | 4",
          "Blue"),
     ],
     "score": one_of("west")},
]

STRATEGIES = {
    "Zero-Shot": {"description": "Just the task and the input."},
    "Few-Shot": {"description": "Task plus 2 worked examples before the input."},
    "Chain-of-Thought": {"description": "Task plus an instruction to reason step by step before answering."},
    "Role + CoT": {"description": "Gives the model an expert role, then asks it to reason step by step."},
}

ANSWER_FORMAT = "Put your final answer after 'ANSWER:' at the end of your reply."
COT_LINE = "Work through it step by step first, writing out your reasoning, then give the final answer."

def build_prompt(strategy_name: str, task: dict):
    # returns (system prompt or None, user message)
    parts = [task["instruction"]]
    if strategy_name == "Few-Shot":
        examples = "\n\n".join(f"Input:\n{i}\nANSWER: {o}" for i, o in task["examples"])
        parts.append("Examples:\n\n" + examples)
    parts.append("Input:\n" + task["input"])
    if strategy_name in ("Chain-of-Thought", "Role + CoT"):
        parts.append(COT_LINE)
    parts.append(ANSWER_FORMAT)
    article = "an" if task["role"][0] in "aeiou" else "a"
    system = f"You are {article} {task['role']}." if strategy_name == "Role + CoT" else None
    return system, "\n\n".join(parts)

def final_answer(text: str) -> str:
    # everything after the last ANSWER:, or the whole reply if the model skipped it
    idx = text.rfind("ANSWER:")
    return text[idx + len("ANSWER:"):].strip() if idx != -1 else text.strip()

# --- calling the model ---

def run_once(client: anthropic.Anthropic, strategy_name: str, task: dict) -> dict:
    system, user = build_prompt(strategy_name, task)
    kwargs = dict(
        model=MODEL,
        max_tokens=8000,
        output_config={"effort": EFFORT},
        messages=[{"role": "user", "content": user}],
    )
    if system:
        kwargs["system"] = system

    start = time.perf_counter()
    response = client.messages.create(**kwargs)
    latency = time.perf_counter() - start

    text = "".join(b.text for b in response.content if b.type == "text")
    answer = final_answer(text)
    # a refusal (the model's safety check declining) scores 0 but is tracked on its own,
    # so it's clear the model said no rather than got it wrong. no fallback model is used,
    # since that would mix two models' results together.
    refused = response.stop_reason == "refusal"
    refusal_category = getattr(response.stop_details, "category", None) if refused else None
    accuracy = 0.0 if refused else float(task["score"](answer))

    usage = response.usage
    cost = usage.input_tokens / 1e6 * PRICE_IN_PER_M + usage.output_tokens / 1e6 * PRICE_OUT_PER_M
    return {
        "strategy": strategy_name,
        "task_id": task["id"],
        "task_type": task["type"],
        "difficulty": task["difficulty"],
        "accuracy": round(accuracy, 4),
        "latency_s": round(latency, 3),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "tokens_used": usage.input_tokens + usage.output_tokens,
        "cost_usd": round(cost, 6),
        "stop_reason": response.stop_reason,
        "refused": refused,
        "refusal_category": refusal_category,
        "answer": answer[:300],
    }


def run_benchmark(verbose: bool = True, repeats: int = 1) -> pd.DataFrame:
    # loop every strategy over every task and collect results
    client = anthropic.Anthropic()
    results = []
    total = len(STRATEGIES) * len(TASKS) * repeats

    print("\n" + "═" * 55)
    print("  LLM PROMPT STRATEGY BENCHMARKER")
    print("═" * 55)
    print(f"  Model      : {MODEL} (effort {EFFORT})")
    print(f"  Strategies : {len(STRATEGIES)}")
    print(f"  Tasks      : {len(TASKS)}")
    print(f"  Total runs : {total}")
    print("═" * 55)

    for strategy_name in STRATEGIES:
        if verbose:
            print(f"\n▶ Running: {strategy_name}")
            print(f"  {STRATEGIES[strategy_name]['description']}")

        for task in TASKS:
            for _ in range(repeats):
                result = run_once(client, strategy_name, task)
                results.append(result)
                if verbose:
                    bar = "█" * int(result["accuracy"] * 20) + "░" * (20 - int(result["accuracy"] * 20))
                    note = f"  REFUSED ({result['refusal_category']})" if result["refused"] else ""
                    print(f"  [{task['id']}] {task['type']:<20} acc={result['accuracy']:.0%}  {bar}  "
                          f"{result['latency_s']:.1f}s  ${result['cost_usd']:.4f}{note}")

    return pd.DataFrame(results)

# --- aggregate the results per strategy ---

def analyze(df: pd.DataFrame) -> pd.DataFrame:
    # roll up per-task results into a single summary row per strategy
    summary = df.groupby("strategy").agg(
        avg_accuracy=("accuracy", "mean"),
        avg_latency=("latency_s", "mean"),
        total_tokens=("tokens_used", "sum"),
        total_cost=("cost_usd", "sum"),
        tasks_run=("task_id", "count"),
        refusals=("refused", "sum"),
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
    subtitle = f"{MODEL}  ·  4 strategies  ·  {len(TASKS)} tasks  ·  {datetime.now().strftime('%Y-%m-%d')}"
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

    im = ax3.imshow(pivot.values, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)

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
    print(f"  {'Rank':<5} {'Strategy':<20} {'Accuracy':>10} {'Latency':>10} {'Efficiency':>12} {'Cost':>12} {'Refused':>9}")
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
            f"{int(row['refusals']):>9}/{int(row['tasks_run'])}"
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
    parser.add_argument("--repeats", type=int, default=1, help="Runs per strategy/task pair")
    parser.add_argument("--output", type=str, default="benchmark_results.png",
                        help="Output path for dashboard image")
    args = parser.parse_args()

    df = run_benchmark(verbose=not args.quiet, repeats=args.repeats)
    summary = analyze(df)
    print_summary_table(summary)

    df.to_csv("benchmark_raw.csv", index=False)
    print(f"[✓] Raw results saved to 'benchmark_raw.csv'")

    visualize(df, summary, output_path=args.output)
    print(f"\nDone. Open '{args.output}' to view the dashboard.\n")
