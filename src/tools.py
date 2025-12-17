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
        "texts": data.data[:500],   # limit for laptop safety
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
