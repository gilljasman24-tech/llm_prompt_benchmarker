# llm_prompt_benchmarker
I wanted to see whether different ways of prompting an LLM make a measurable
difference on business tasks. This runs four prompt strategies (zero-shot, few-shot,
chain-of-thought, and role + chain-of-thought) across ten tasks with known right
answers, calls a real model for every run, checks each answer automatically, and
records accuracy, speed, and cost. Results go to a CSV and a 4-panel dashboard.

The tasks are small but real: summarizing an earnings update and a policy, classifying
a complaint and a support ticket, pulling fields out of a messy invoice and a lease,
reading a review, writing a SQL query (which gets run against a tiny SQLite database
and has to return the right customer), finding the root cause in server logs, and
spotting which region's conversion rate dropped most.

## Results (Claude Opus 5.5, effort low, 3 runs per strategy/task, 120 calls, about $0.67)

| Strategy | Accuracy | Avg latency | Cost | Refused |
|---|---|---|---|---|
| Zero-shot | 100% | 2.4s | $0.12 | 0/30 |
| Few-shot | 100% | 2.5s | $0.15 | 0/30 |
| Chain-of-thought | 73% | 3.8s | $0.20 | 8/30 |
| Role + chain-of-thought | 70% | 3.7s | $0.20 | 9/30 |

What I found:
- **Every answer the model actually gave was right.** These tasks are too easy for a
  current model to separate the strategies on accuracy. Harder tasks are the next step.
- **Asking it to "write out your reasoning step by step" backfired.** About 28% of those
  runs were declined by the model's safety check (category `reasoning_extraction`),
  even on harmless tasks like summarizing. Newer models already reason internally, and
  asking them to print that reasoning can trip this check.
- **Chain-of-thought cost about 70% more and took about 55% longer** than zero-shot,
  for no accuracy gain here.
- So on this model, the plain prompt or a couple of examples did best.

Refusals are tracked in their own column so a "no" isn't mixed up with a wrong answer.
Small sample, so treat this as a first look, not a final answer.

## Run it
```
pip install anthropic pandas matplotlib
export ANTHROPIC_API_KEY=...
python llm_benchmark.py --repeats 3
```
