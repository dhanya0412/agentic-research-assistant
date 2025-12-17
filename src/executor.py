from tools import load_text_classification_dataset, preprocess_texts
from planner import classify_task_intent

def execute_subtask(subtask: dict, memory: dict):
    task_text = subtask["task"]
    intent = classify_task_intent(task_text)

    # 1️⃣ LOAD DATASET
    if intent == "LOAD_DATASET":
        if "dataset" in memory:
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Dataset already loaded"
            }

        dataset = load_text_classification_dataset()
        memory["dataset"] = dataset

        return {
            "task_id": subtask["id"],
            "status": "success",
            "output": f"Loaded dataset with {len(dataset['texts'])} samples"
        }

    # 2️⃣ PREPROCESS DATA
    if intent == "PREPROCESS_DATA":
        if "dataset" not in memory:
            return {
                "task_id": subtask["id"],
                "status": "failed",
                "output": "Dataset not available for preprocessing"
            }

        if "features" in memory:
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Data already preprocessed"
            }

        features = preprocess_texts(memory["dataset"]["texts"])
        memory["features"] = features

        return {
            "task_id": subtask["id"],
            "status": "success",
            "output": "Text data preprocessed into features"
        }

    # 3️⃣ OTHER TASKS (not implemented yet)
    return {
        "task_id": subtask["id"],
        "status": "unsupported",
        "output": f"Intent classified as {intent}"
    }

def execute_plan(plan: dict):
    memory = {}
    results = []

    for subtask in plan["subtasks"]:
        result = execute_subtask(subtask, memory)
        results.append(result)

    return results, memory
