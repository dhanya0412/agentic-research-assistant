import numpy as np

#naive bayes classifer using numpy for classification into various classes

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_log_prior = None
        self.feature_log_prob = None
        self.classes = None

    def fit(self, X, y):
        """
        X: (n_samples, n_features) count matrix
        y: (n_samples,) labels
        """
        X = np.array(X)
        y = np.array(y)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            class_count[idx] = X_c.shape[0]
            feature_count[idx] = X_c.sum(axis=0)

        #log priors
        self.class_log_prior = np.log(class_count / class_count.sum())

        #log likelihoods with Laplace smoothing
        smoothed_fc = feature_count + 1
        smoothed_totals = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob = np.log(smoothed_fc / smoothed_totals)

    def predict(self, X):
        X = np.array(X)
        log_probs = X @ self.feature_log_prob.T + self.class_log_prior
        return self.classes[np.argmax(log_probs, axis=1)]
