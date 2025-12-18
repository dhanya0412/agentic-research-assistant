import os
import fitz 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def load_text_classification_dataset():
    """
    Loads a real text classification dataset.
    """
    data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes")
    )

    return {
        "texts": data.data[:500],  
        "labels": data.target[:500],
        "label_names": data.target_names
    }


def preprocess_texts(texts):
    """
    Converts raw text into TF-IDF vectors.
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X = vectorizer.fit_transform(texts)

    return {
        "features": X,
        "vocab_size": len(vectorizer.vocabulary_)
    }

def evaluate_text_classifier(memory):
    """
    Evaluates a trained text classification model using a simple train-test split.
    """
    if "model" not in memory or "features" not in memory or "dataset" not in memory:
        raise ValueError("Missing model, features, or dataset for evaluation")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = memory["features"]["features"].toarray()
    y = memory["dataset"]["labels"]
    model = memory["model"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    memory["observations"] = {
        "baseline_accuracy": round(float(accuracy), 4),
        "model": "Multinomial Naive Bayes",
        "features": "Bag-of-Words (5000 vocab)",
        "dataset": "20 Newsgroups (subset)",
        "conclusion": (
            "The dataset exhibits strong topical separability. "
            "Even a simple probabilistic baseline achieves meaningful performance."
        )
    }

    return f"Baseline evaluation completed with accuracy={accuracy:.4f}"

def summarize_texts(texts, client, max_docs=3, chunk_size=1500):
    """
    Summarize documents safely by chunking long texts.
    """
    summaries = []

    for i, text in enumerate(texts[:max_docs]):
        text = text.strip()
        if not text:
            continue

        # Split text into chunks
        chunks = [
            text[j:j + chunk_size]
            for j in range(0, len(text), chunk_size)
        ]

        chunk_summaries = []

        for idx, chunk in enumerate(chunks[:5]):  # limit chunks per doc
            prompt = f"""
Summarize the following part of a research paper.
Focus on key ideas only.

Text:
{chunk}
"""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            chunk_summaries.append(
                response.choices[0].message.content.strip()
            )

        # Combine chunk summaries
        combined_summary = " ".join(chunk_summaries)

        summaries.append({
            "doc_id": i,
            "summary": combined_summary
        })

    return summaries


def load_documents_from_folder(folder_path="documents"):
    """
    Loads text documents from a local folder.
    """
    texts = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found")

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            doc = fitz.open(pdf_path)

            full_text = ""
            for page in doc:
                full_text += page.get_text()

            if full_text.strip():
                texts.append(full_text)

    return {
        "texts": texts,
        "source": folder_path,
        "num_documents": len(texts)
    }

