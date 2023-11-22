import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.euclidean_distance(
            x, x_train) for x_train in self.X_train]
        k_nearest = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest]
        result = np.bincount(k_nearest_labels).argmax()
        return result
