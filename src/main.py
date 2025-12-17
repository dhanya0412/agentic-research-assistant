import logging
from planner import generate_plan
from executor import execute_plan

logging.basicConfig(
    filename="logs/agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    goal = "Build a text classification system using machine learning"

    plan = generate_plan(goal)
    results, memory = execute_plan(plan)

    print("\nExecution Results:")
    for r in results:
        print(r)

    print("\nFinal Memory Keys:")
    print(memory.keys())
