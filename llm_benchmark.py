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