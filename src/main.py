import logging
import argparse
from planner import generate_plan
from executor import execute_plan
from research_mode import infer_research_mode


logging.basicConfig(
    filename="logs/agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Agentic Research Assistant"
)

    parser.add_argument(
        "--goal",
        type=str,
        required=False,
        help="Research goal for the agent"
    )
    args = parser.parse_args()

    if args.goal:
        goal = args.goal
    else:
        print("\nChoose a research goal:")
        print("1. Build a machine learning text classification model")
        print("2. Summarize recent research papers on transformers")

        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            goal = "Build a machine learning text classification model"
        elif choice == "2":
            goal = "Summarize recent research papers on transformers"
        else:
            print("Invalid choice. Exiting.")
            exit(1)


    mode = infer_research_mode(goal)
    print("Research mode:", mode)

    plan = generate_plan(goal)
    memory = {"research_mode": mode}
    print("\nGenerated Plan:")
    print("-" * 40)

    for step in plan["subtasks"]:
        print(f"Step {step['id']}: {step['task']}")

    print("-" * 40)

    results, memory = execute_plan(plan, memory)

    print("\nExecution Results:")
    for r in results:
        print(r)

    print("\nFinal Memory Keys:")
    print(memory.keys())

    if "summaries" in memory:
        print("\nGenerated Summaries:\n" + "-" * 40)

        for item in memory["summaries"]:
            print(f"\nDocument {item['doc_id'] + 1} Summary:")
            print(item["summary"])
            print("-" * 40)
