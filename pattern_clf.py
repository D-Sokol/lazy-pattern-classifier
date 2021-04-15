from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
import random
import numpy as np
import pandas as pd


class LazyPatternClassifier(BaseEstimator, ClassifierMixin):
    weights_strategies = ['uniform', 'from_objects', 'from_classifiers']

    def __init__(self, tolerance=0.0, use_softmax=True, weights_strategy='uniform', weight_classifiers=True, weights_iters=5):
        self.tolerance = tolerance
        self.use_softmax = use_softmax
        if weights_strategy not in self.weights_strategies:
            raise ValueError("Unknown weights strategy: {}. Allowed only {}".format(weights_strategy, self.weights_strategies))
        self.weights_strategy = weights_strategy
        self.weight_classifiers = weight_classifiers
        self.weights_iters = weights_iters


    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.values.astype(bool)
        Xnum = X.select_dtypes(include=float).values
        Xcat = X.select_dtypes(exclude=float).values
        self.Xnum_p = Xnum[y]
        self.Xnum_n = Xnum[~y]
        self.Xcat_p = Xcat[y]
        self.Xcat_n = Xcat[~y]
        self._set_weights(Xnum, Xcat, y)


    def _set_weights(self, Xnum, Xcat, y):
        if self.weights_strategy == 'uniform':
            self.weights_p = np.ones(self.Xnum_p.shape[0])
            self.weights_n = np.ones(self.Xnum_n.shape[0])
        else:
            if self.weights_strategy == 'from_classifiers':
                # TODO: implement alternative strategy
                raise NotImplementedError

            objects_weights = np.full(len(Xnum), 1/len(Xnum))
            for _ in range(self.weights_iters):
                eps_min = float('inf')
                eps_min_ix = None  # FIXME: unused variable

                for ix_clf in range(y.size):
                    y_pred = np.empty(y.size, dtype=bool)
                    next_opposite_index = 0
                    for ix_obj in range(y.size):
                        pattern = self._get_pattern(Xnum[ix_clf], Xnum[ix_obj], Xcat[ix_clf], Xcat[ix_obj])
                        other_num = (self.Xnum_n if y[ix_clf] else self.Xnum_p)
                        other_cat = (self.Xcat_n if y[ix_clf] else self.Xcat_p)
                        mask = self._satisfy(*pattern, other_num, other_cat)
                        if y[ix_clf] != y[ix_obj]:
                            assert np.array_equal(Xnum[ix_obj], other_num[next_opposite_index])
                            assert mask[next_opposite_index]
                            # Exclude the classified object from consideration
                            mask[next_opposite_index] = False
                            next_opposite_index += 1
                        y_pred[ix_obj] = mask.any() ^ y[ix_clf]
                    epsilon = objects_weights[y_pred != y].sum()
                    if epsilon < eps_min:
                        eps_min = epsilon
                        eps_min_ix = ix_clf

                if eps_min >= 0.5:
                    break

                alpha = np.log(- 1 + 1 / eps_min) / 2
                # TODO: alternative strategy: alphas[ix] += alpha
                objects_weights *= np.where(y_pred == y, np.exp(-alpha), np.exp(+alpha))
                objects_weights /= objects_weights.sum()
            self.weights_p = objects_weights[y]
            self.weights_n = objects_weights[~y]


    def predict(self, X: pd.DataFrame):
        y_pred = np.empty(X.shape[0], dtype=bool)
        Xnum = X.select_dtypes(include=float).values
        Xcat = X.select_dtypes(exclude=float).values
        for i in range(X.shape[0]):
            y_pred[i] = self._predict_one(Xnum[i], Xcat[i])
        return y_pred


    def predict_proba(self, X: pd.DataFrame):
        result = np.empty((X.shape[0], 2))
        Xnum = X.select_dtypes(include=float).values
        Xcat = X.select_dtypes(exclude=float).values
        for i, (num, cat) in enumerate(zip(Xnum, Xcat)):
            result[i] = (self._score(num, cat, False), self._score(num, cat, True))

        if self.use_softmax:
            softmax(result, copy=False)
        else:
            s = result.sum(axis=1, keepdims=True)
            mask = (s.squeeze() != 0)
            result[mask] /= s[mask]
            result[~mask] = 0.5
        return result


    def _predict_one(self, num: np.ndarray, cat: np.ndarray) -> bool:
        return self._score(num, cat, True) > self._score(num, cat, False)


    @staticmethod
    def _get_pattern(num1, num2, cat1, cat2):
        pattern_mins = np.minimum(num1, num2)
        pattern_maxs = np.maximum(num1, num2)
        pattern_cats = np.stack((cat1, cat2))
        return (pattern_mins, pattern_maxs, pattern_cats)


    @staticmethod
    def _satisfy(pattern_mins, pattern_maxs, pattern_cats, other_num, other_cat):
        mask = ((other_num >= pattern_mins) & (other_num <= pattern_maxs)).all(axis=1)
        for i_attr in range(pattern_cats.shape[1]):
            mask &= np.isin(other_cat[:, i_attr], pattern_cats[:, i_attr])
        return mask


    def _score(self, num, cat, to_positive=True):
        this_num  = self.Xnum_p if to_positive else self.Xnum_n
        other_num = self.Xnum_n if to_positive else self.Xnum_p
        this_cat  = self.Xcat_p if to_positive else self.Xcat_n
        other_cat = self.Xcat_n if to_positive else self.Xcat_p
        this_w    = self.weights_p if to_positive else self.weights_n
        other_w   = self.weights_n if to_positive else self.weights_p

        votes = 0
        for bnum, bcat, weight in zip(this_num, this_cat, this_w):
            pattern = self._get_pattern(num, bnum, cat, bcat)
            mask = self._satisfy(*pattern, other_num, other_cat)

            if self.weight_classifiers:
                if mask.mean() <= self.tolerance:
                    votes += weight
            else:
                if mask @ other_w <= self.tolerance:
                    votes += 1

        return votes / len(this_num)

