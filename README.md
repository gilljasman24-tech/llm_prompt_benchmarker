# llm_prompt_benchmarker
I wanted to see whether different ways of prompting an LLM make a measurable
difference on business tasks. This is the framework for testing that: it runs four
prompt strategies across ten tasks (summarization, classification, data extraction,
reasoning, etc.), scores each one on accuracy, speed, and cost, saves everything to a
CSV, and builds a 4-panel dashboard.

**Heads up: the results right now are simulated, not real.** There are no API calls yet.
Each strategy's accuracy, latency, and cost come from numbers set in the code plus some
random noise, so the dashboard shows how the pipeline works, not what real models do.

Next step: swap `simulate_response()` for real model calls and score the actual answers.
