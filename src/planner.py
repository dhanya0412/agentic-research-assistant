import json
import logging
from groq import Groq
from config import GROQ_API_KEY


client = Groq(api_key=GROQ_API_KEY)


def build_planner_prompt(goal: str) -> str:
    return f"""
You are a planning agent.

Your job is to break a goal into clear, ordered subtasks.

Rules:
- Return ONLY valid JSON
- No explanations outside JSON
- No markdown
- No extra text

JSON schema:
{{
  "goal": string,
  "subtasks": [
    {{
      "id": number,
      "task": string,
      "reason": string
    }}
  ]
}}

Goal:
{goal}
"""


def validate_plan_schema(plan: dict):
    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dictionary")

    if "goal" not in plan or "subtasks" not in plan:
        raise ValueError("Missing required fields")

    if not isinstance(plan["subtasks"], list):
        raise ValueError("Subtasks must be a list")

    for subtask in plan["subtasks"]:
        if not all(k in subtask for k in ("id", "task", "reason")):
            raise ValueError("Each subtask must contain id, task, and reason")


def generate_plan(goal: str):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": build_planner_prompt(goal)}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content

    try:
        plan = json.loads(raw_output)
        validate_plan_schema(plan)
        return plan
    except json.JSONDecodeError:
        logging.error("Planner returned invalid JSON")
        logging.error(raw_output)
        raise ValueError("Invalid planner output")

def classify_task_intent(task: str) -> str:
    """
    Converts a natural language task into a fixed intent label.
    """

    prompt = f"""
You are an intent classifier for an AI agent.

Classify the task into ONE of the following intents:
- LOAD_DATASET
- PREPROCESS_DATA
- TRAIN_MODEL
- EVALUATE_MODEL
- OTHER

Return ONLY the intent name. No explanation.

Task:
{task}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
