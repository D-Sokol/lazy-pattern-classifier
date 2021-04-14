from sklearn.base import BaseEstimator, ClassifierMixin
import random
import numpy as np
import pandas as pd


class LazyPatternClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, tolerance=0.0):
        self.tolerance = tolerance


    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.values.astype(bool)
        Xnum = X.select_dtypes(include=float).values
        Xcat = X.select_dtypes(exclude=float).values
        self.Xnum_p = Xnum[y]
        self.Xnum_n = Xnum[~y]
        self.Xcat_p = Xcat[y]
        self.Xcat_n = Xcat[~y]
        self.weights = None  # TODO


    def predict(self, X: pd.DataFrame):
        y_pred = np.empty(X.shape[0], dtype=bool)
        Xnum = X.select_dtypes(include=float).values
        Xcat = X.select_dtypes(exclude=float).values
        for i in range(X.shape[0]):
            y_pred[i] = self._predict_one(Xnum[i], Xcat[i])
        return y_pred


    def _predict_one(self, num: np.ndarray, cat: np.ndarray) -> bool:
        return self._score(num, cat, True) > self._score(num, cat, False)


    def _score(self, num, cat, to_positive=True):
        this_num  = self.Xnum_p if to_positive else self.Xnum_n
        other_num = self.Xnum_n if to_positive else self.Xnum_p
        this_cat  = self.Xcat_p if to_positive else self.Xcat_n
        other_cat = self.Xcat_n if to_positive else self.Xcat_p

        votes = 0
        for bnum, bcat in zip(this_num, this_cat):
            pattern_mins = np.minimum(num, bnum)
            pattern_maxs = np.maximum(num, bnum)
            pattern_cats = np.stack((cat, bcat))

            mask = ((other_num >= pattern_mins) & (other_num <= pattern_maxs)).all(axis=1)
            for i_attr in range(pattern_cats.shape[1]):
                mask &= np.isin(other_cat[:, i_attr], pattern_cats[:, i_attr])

            if mask.mean() <= self.tolerance:
                votes += 1

        # TODO: weights, coefficients, other
        return votes / len(this_num)

