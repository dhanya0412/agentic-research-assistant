from tools import evaluate_text_classifier, load_text_classification_dataset, preprocess_texts, summarize_texts, load_documents_from_folder
from planner import classify_task_intent, client
from models import MultinomialNaiveBayes


def execute_subtask(subtask: dict, memory: dict):
    task_text = subtask["task"]
    intent = classify_task_intent(task_text)

    if intent == "LOAD_DATASET":
        if "dataset" in memory:
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Dataset already loaded"
        }

        if memory.get("research_mode") == "TEXT_SUMMARIZATION":
            dataset = load_documents_from_folder()
            memory["dataset"] = dataset
            return {
                "task_id": subtask["id"],
                "status": "success",
            "output": f"Loaded {dataset['num_documents']} PDF documents for summarization"
        }

    # TEXT_CLASSIFICATION (default)
        dataset = load_text_classification_dataset()
        memory["dataset"] = dataset
        return {
            "task_id": subtask["id"],
            "status": "success",
            "output": f"Loaded dataset with {len(dataset['texts'])} samples"
    }

    if intent == "PREPROCESS_DATA":
        if memory.get("research_mode") == "TEXT_SUMMARIZATION":
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Preprocessing not applicable for summarization"
        }

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
    

    #use the naive bayes model for classification
    if intent == "TRAIN_MODEL":
        if memory.get("research_mode") != "TEXT_CLASSIFICATION":
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Training not applicable for this research mode"
            }
        if "features" not in memory:
            return {
                "task_id": subtask["id"],
                "status": "failed",
                "output": "Features not available for training"
            }

        if "model" in memory:
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Model already trained"
            }

        X = memory["features"]["features"].toarray()
        y = memory["dataset"]["labels"]

        model = MultinomialNaiveBayes()
        model.fit(X, y)

        memory["model"] = model

        return {
            "task_id": subtask["id"],
            "status": "success",
            "output": "Naive Bayes model trained successfully"
        }
    
    if intent == "EVALUATE_MODEL":
        if memory.get("research_mode") != "TEXT_CLASSIFICATION":
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Evaluation not applicable for this research mode"
            }

        output = evaluate_text_classifier(memory)
        return {
            "task_id": subtask["id"],
            "status": "skipped",
            "output": output
        }
    
    
    if intent == "SUMMARIZE_TEXT":
       
        if "dataset" not in memory:
            if memory.get("research_mode") == "TEXT_SUMMARIZATION":
                dataset = load_documents_from_folder()
                memory["dataset"] = dataset
            else:
                return {
                    "task_id": subtask["id"],
                    "status": "failed",
                    "output": "No dataset available for summarization"
                }

        if "summaries" in memory:
            return {
                "task_id": subtask["id"],
                "status": "skipped",
                "output": "Summaries already generated"
            }

        summaries = summarize_texts(
            memory["dataset"]["texts"],
            client
        )

        memory["summaries"] = summaries
        memory.setdefault("observations", {})["num_summaries"] = len(summaries)

        return {
            "task_id": subtask["id"],
            "status": "success",
            "output": f"Generated summaries for {len(summaries)} documents"
        }


    return {
        "task_id": subtask["id"],
        "status": "unsupported",
        "output": f"Intent classified as {intent}"
    }

def execute_plan(plan: dict, memory=None):
    if memory is None:
        memory = {}

    results = []

    for subtask in plan["subtasks"]:
        result = execute_subtask(subtask, memory)
        results.append(result)

    return results, memory

