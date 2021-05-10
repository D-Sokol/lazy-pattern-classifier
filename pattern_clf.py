from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
import numpy as np
import pandas as pd


class LazyPatternClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tolerance=0.0, weight_classifiers=True,
                 weights_iters=0):
                 #weights_iters=0, temperature=1.):
        self.tolerance = tolerance
        self.weight_classifiers = weight_classifiers
        self.weights_iters = weights_iters
        #self.temperature = temperature
        self.temperature = 1.0
        self.Xnum_p = self.Xnum_n = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.values.astype(bool)
        Xnum = X.values
        assert X.select_dtypes(exclude=float).values.size == 0
        self.Xnum_p = Xnum[y]
        self.Xnum_n = Xnum[~y]
        self._set_weights(Xnum, y)

    def _set_weights(self, Xnum, y):
        y_pred = np.empty((y.size, y.size), dtype=bool)
        for ix_clf in range(y.size):
            next_opposite_index = 0
            for ix_obj in range(y.size):
                pattern = self._get_pattern(Xnum[ix_clf], Xnum[ix_obj])
                other_num = (self.Xnum_n if y[ix_clf] else self.Xnum_p)
                mask = self._satisfy(*pattern, other_num)
                if y[ix_clf] != y[ix_obj]:
                    # assert np.array_equal(Xnum[ix_obj], other_num[next_opposite_index])
                    # assert mask[next_opposite_index]
                    # Exclude the classified object from consideration
                    mask[next_opposite_index] = False
                    next_opposite_index += 1

                # Since there are no weights yet, we use simple average
                y_pred[ix_clf, ix_obj] = (mask.mean() <= self.tolerance) ^ y[ix_clf]

        objects_weights = np.full(y.size, 1/y.size)
        for _ in range(self.weights_iters):
            eps_min = float('inf')

            epsilons = np.where(y_pred != y, objects_weights, 0.).sum(axis=1)
            ix_best = epsilons.argmin()
            eps_min = epsilons[ix_best]

            if eps_min >= 0.5:
                break

            alpha = np.log(- 1 + 1 / max(eps_min, 1e-6)) / 2
            objects_weights *= np.where(y_pred[ix_best] == y, np.exp(-alpha), np.exp(+alpha))
            objects_weights /= objects_weights.sum()

        self.weights_p = objects_weights[y] / self.temperature
        self.weights_n = objects_weights[~y] / self.temperature
        #softmax(self.weights_p[None], copy=False)
        #softmax(self.weights_n[None], copy=False)

    def predict(self, X: pd.DataFrame):
        y_pred = np.empty(X.shape[0], dtype=bool)
        Xnum = X.select_dtypes(include=float).values
        assert X.select_dtypes(exclude=float).values.size == 0
        for i in range(X.shape[0]):
            y_pred[i] = self._predict_one(Xnum[i])
        return y_pred

    def predict_proba(self, X: pd.DataFrame):
        result = np.empty((X.shape[0], 2))
        Xnum = X.select_dtypes(include=float).values
        assert X.select_dtypes(exclude=float).values.size == 0
        for i, num in enumerate(Xnum):
            result[i] = (self._score(num, False), self._score(num, True))

        softmax(result, copy=False)
        return result

    def _predict_one(self, num: np.ndarray) -> bool:
        return self._score(num, True) > self._score(num, False)

    @staticmethod
    def _get_pattern(num1, num2):
        pattern_mins = np.minimum(num1, num2)
        pattern_maxs = np.maximum(num1, num2)
        return pattern_mins, pattern_maxs

    @staticmethod
    def _satisfy(pattern_mins, pattern_maxs, other_num):
        mask = np.logical_and((other_num >= pattern_mins), (other_num <= pattern_maxs)).all(axis=1)
        return mask

    def _score(self, num, to_positive=True):
        this_num = self.Xnum_p if to_positive else self.Xnum_n
        other_num = self.Xnum_n if to_positive else self.Xnum_p
        this_w = self.weights_p if to_positive else self.weights_n
        other_w = self.weights_n if to_positive else self.weights_p

        votes = 0
        for bnum, weight in zip(this_num, this_w):
            pattern = self._get_pattern(num, bnum)
            mask = self._satisfy(*pattern, other_num)

            if self.weight_classifiers:
                if mask.mean() <= self.tolerance:
                    votes += weight
            else:
                if mask @ other_w <= self.tolerance:
                    votes += 1

        return votes
