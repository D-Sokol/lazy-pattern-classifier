from functools import wraps
import os
import pandas as pd


def _read_csv(name, **kw):
    path = os.path.join(os.path.dirname(__file__), 'data', name)
    return pd.read_csv(path, **kw)


def apply_one_hot(fun):
    @wraps(fun)
    def wrapper(*args, **kw):
        X, y = fun(*args, **kw)
        X = pd.get_dummies(X, dtype=float)
        return X, y
    return wrapper


@apply_one_hot
def get_breast_cancer():
    df = _read_csv('breast-cancer-wisconsin.zip', index_col=0)
    df.drop('Unnamed: 32', axis=1, inplace=True)
    X, y = df.drop('diagnosis', axis=1), df['diagnosis'] == 'M'
    return X, y


@apply_one_hot
def get_heart_disease():
    df = _read_csv('heart-disease-uci.zip')
    X, y = df.drop('target', axis=1), df['target']
    X = X.astype(float, copy=False)
    return X, y


@apply_one_hot
def get_mammographic_mass():
    df = _read_csv('mammographic-mass.zip')
    X, y = df.drop('Severity', axis=1), df['Severity']
    X = X.astype(float, copy=False)
    return X, y


@apply_one_hot
def get_seismic_bumps():
    df = _read_csv('seismic-bumps.zip')
    X, y = df.drop('class', axis=1), df['class']
    X.drop(['nbumps6', 'nbumps7', 'nbumps89'], axis=1, inplace=True)
    X['seismic'] = (X['seismic'] == 'a')
    X['shift'] = (X['shift'] == 'N')
    X = X.astype({col: float for col in X if col not in ('seismoacoustic', 'hazard')}, copy=False)
    return X, y


@apply_one_hot
def get_titanic():
    df = _read_csv('titanic.zip', index_col=0)
    X, y = df.drop('Survived', axis=1), df['Survived']
    X.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    X['Sex'] = (X['Sex'] == 'male')
    X.fillna({'Age': X['Age'].median(), 'Embarked': 'X'}, inplace=True)
    X = X.astype({col: float for col in X if col not in ('Embarked',)}, copy=False)
    return X, y

def make_GSE_getter(name):
    def getter():
        df = _read_csv(name, index_col=0)
        X, y = df.drop('type', axis=1), df['type']
        y = (y == 'normal')
        assert X.values.dtype.kind == 'f'
        return X, y
    return getter


get_breast_GSE = make_GSE_getter('Breast_GSE70947.csv')
get_liver_GSE = make_GSE_getter('Liver_GSE14520_U133A.csv')
get_prostate_GSE = make_GSE_getter('Prostate_GSE6919_U95B.csv')



def get_hiv():
    df = _read_csv('HIV.csv')
    X, y = df[['smiles']], df['HIV_active']
    return X, y
