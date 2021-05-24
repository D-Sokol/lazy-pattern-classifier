from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
import pandas as pd


class LazyPatternClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_positive=True):
        self.use_positive = use_positive

    def fit(self, X, y):
        Xnum, y = check_X_y(X, y)

        self.n_features_in_ = Xnum.shape[-1]
        self.classes_ = unique_labels(y)
        if self.classes_.size > 2:
            raise ValueError

        y = (y == self.classes_[-1])
        self.Xnum_p_ = Xnum[y]
        self.Xnum_n_ = Xnum[~y]
        self._set_weights(Xnum, y)
        return self

    def _set_weights(self, Xnum, y):
        pass

    def predict(self, X):
        check_is_fitted(self, ['Xnum_p_', 'Xnum_n_'])
        Xnum = check_array(X)

        if self.classes_.size == 1:
            return np.full(Xnum.shape[0], self.classes_[0])

        y_pred = np.empty(Xnum.shape[0], dtype=int)
        for i in range(Xnum.shape[0]):
            y_pred[i] = self._predict_one(Xnum[i])
        return self.classes_[y_pred]

    def _predict_one(self, num: np.ndarray) -> bool:
        clfs = self.Xnum_p_ if self.use_positive else self.Xnum_n_
        objs = self.Xnum_n_ if self.use_positive else self.Xnum_p_
        for p_clf in clfs:
            pattern = self._get_pattern(p_clf, num)
            if not self._satisfy(*pattern, objs).any():
                return self.use_positive
        return not self.use_positive

    def _more_tags(self):
        return {'binary_only': True,}

    @staticmethod
    def _get_pattern(num1, num2):
        pattern_mins = np.minimum(num1, num2)
        pattern_maxs = np.maximum(num1, num2)
        return pattern_mins, pattern_maxs

    @staticmethod
    def _satisfy(pattern_mins, pattern_maxs, other_num):
        mask = np.logical_and((other_num >= pattern_mins), (other_num <= pattern_maxs)).all(axis=1)
        return mask
