from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
import pandas as pd


class LazyPatternClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_positive=True, alpha=0.0, beta=0.0):
        self.use_positive = use_positive
        self.alpha = alpha
        self.beta = beta

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
        this_num = self.Xnum_p_ if self.use_positive else self.Xnum_n_
        other_num = self.Xnum_n_ if self.use_positive else self.Xnum_p_
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
        votes = 0
        for p_clf, clf_weight in zip(clfs, self.weights_):
            pattern = self._get_pattern(p_clf, num)
            if self._satisfy(*pattern, objs).mean() <= self.alpha:
                votes += clf_weight
            else:
                votes -= clf_weight
        return self.use_positive == (votes > self.beta)

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
