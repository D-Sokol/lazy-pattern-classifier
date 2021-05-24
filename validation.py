#!/usr/bin/env python3

from sklearn.utils.estimator_checks import check_estimator

from pattern_clf import LazyPatternClassifier


if __name__ == '__main__':
    clf = LazyPatternClassifier()
    check_estimator(clf)
