from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
import numpy as np
import pandas as pd


class LazyPatternClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_positive=True, alpha=0.0, beta=0.0):
        self.use_positive = use_positive
        self.alpha = alpha
        self.beta = beta
        self.Xnum_p = self.Xnum_n = None
        self.weights_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.values.astype(bool)
        Xnum = X.values
        assert X.select_dtypes(exclude=float).values.size == 0
        self.Xnum_p = Xnum[y]
        self.Xnum_n = Xnum[~y]
        self._set_weights(Xnum, y)

    def _set_weights(self, Xnum, y):
        this_num = self.Xnum_p if self.use_positive else self.Xnum_n
        other_num = self.Xnum_n if self.use_positive else self.Xnum_p
        this_n, other_n = len(this_num), len(other_num)
        n = y.size
        prediction_errors = np.empty((this_n, n), dtype=bool)
        for ix_clf in range(this_n):
            for ix_obj in range(n):
                pattern = self._get_pattern(this_num[ix_clf], Xnum[ix_obj])
                mask = self._satisfy(*pattern, other_num)
                prediction_errors[ix_clf, ix_obj] = ((mask.mean() < self.alpha) == self.use_positive)
        prediction_errors ^= y

        objects_weights = np.full(n, 1 / n)
        classifiers_weights = np.zeros(this_n)
        classifiers_is_used = np.zeros(this_n, dtype=bool)
        for _ in range(this_n):
            epsilons = np.where(prediction_errors, objects_weights, 0.).sum(axis=1)
            ix_best = np.ma.masked_array(epsilons, mask=classifiers_is_used).argmin()
            classifiers_is_used[ix_best] = True
            eps_min = epsilons[ix_best]
            if eps_min >= 0.5:
                break

            alpha = np.log(- 1 + 1 / max(eps_min, 1e-6)) / 2
            classifiers_weights[ix_best] = alpha
            objects_weights *= np.where(prediction_errors[ix_best], np.exp(+alpha), np.exp(-alpha))
            objects_weights /= objects_weights.sum()
        self.weights_ = classifiers_weights

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
        clfs = self.Xnum_p if self.use_positive else self.Xnum_n
        objs = self.Xnum_n if self.use_positive else self.Xnum_p
        count_not_falsified = 0
        for p_clf, clf_weight in zip(clfs, self.weights_):
            pattern = self._get_pattern(p_clf, num)
            if self._satisfy(*pattern, objs).mean() <= self.alpha:
                count_not_falsified += clf_weight
        return self.use_positive == (count_not_falsified > self.beta)

    @staticmethod
    def _get_pattern(num1, num2):
        pattern_mins = np.minimum(num1, num2)
        pattern_maxs = np.maximum(num1, num2)
        return pattern_mins, pattern_maxs

    @staticmethod
    def _satisfy(pattern_mins, pattern_maxs, other_num):
        mask = np.logical_and((other_num >= pattern_mins), (other_num <= pattern_maxs)).all(axis=1)
        return mask

    # deprecated
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

        return votes / len(this_num)
