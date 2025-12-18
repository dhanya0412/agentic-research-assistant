from planner import client

def infer_research_mode(goal: str) -> str:
    """
    High-level research intent classification.
    """

    prompt = f"""
Classify the following research goal into ONE category.

Allowed categories:
- TEXT_CLASSIFICATION
- TEXT_SUMMARIZATION
- DATA_EXPLORATION

Return ONLY the category name.

Goal:
{goal}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()
