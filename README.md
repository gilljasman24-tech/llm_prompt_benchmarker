# llm_prompt_benchmarker
I wanted to see whether different ways of prompting an LLM actually make a measurable 
difference on real business tasks. So I built a benchmarker that runs four prompt strategies 
across ten tasks like summarization, classification, data extraction, and reasoning, 
then scores each one on accuracy, speed, and cost. Everything gets saved to a CSV and 
visualized in a dashboard. I found out Role + Chain-of-Thought wins on accuracy but Zero-Shot 
is way more efficient if you're watching costs.