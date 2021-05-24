from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
import numpy as np
import pandas as pd


class LazyPatternClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_positive=True):
        self.use_positive = use_positive
        self.Xnum_p = self.Xnum_n = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.values.astype(bool)
        Xnum = X.values
        assert X.select_dtypes(exclude=float).values.size == 0
        self.Xnum_p = Xnum[y]
        self.Xnum_n = Xnum[~y]
        self._set_weights(Xnum, y)

    def _set_weights(self, Xnum, y):
        pass

    def predict(self, X: pd.DataFrame):
        y_pred = np.empty(X.shape[0], dtype=bool)
        Xnum = X.select_dtypes(include=float).values
        assert X.select_dtypes(exclude=float).values.size == 0
        for i in range(X.shape[0]):
            y_pred[i] = self._predict_one(Xnum[i])
        return y_pred

    def _predict_one(self, num: np.ndarray) -> bool:
        clfs = self.Xnum_p if self.use_positive else self.Xnum_n
        objs = self.Xnum_n if self.use_positive else self.Xnum_p
        for p_clf in clfs:
            pattern = self._get_pattern(p_clf, num)
            if not self._satisfy(*pattern, objs).any():
                return self.use_positive
        return not self.use_positive

    @staticmethod
    def _get_pattern(num1, num2):
        pattern_mins = np.minimum(num1, num2)
        pattern_maxs = np.maximum(num1, num2)
        return pattern_mins, pattern_maxs

    @staticmethod
    def _satisfy(pattern_mins, pattern_maxs, other_num):
        mask = np.logical_and((other_num >= pattern_mins), (other_num <= pattern_maxs)).all(axis=1)
        return mask
