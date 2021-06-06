import itertools
import networkx as nx
import networkx.algorithms.isomorphism as nxiso
import numpy as np
from pysmiles import read_smiles
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from pattern_clf import LazyPatternClassifier


def subchains(G, prefix=(), limit=3):
    if len(prefix) == limit:
        yield prefix
    elif not prefix:
        for root in G:
            yield from subchains(G, (root,), limit=limit)
    else:
        for next_node in G[prefix[-1]]:
            if next_node not in prefix:
                yield from subchains(G, prefix+(next_node,), limit=limit)


def description(mol):
    mapping = mol.nodes(data='element')
    return {
        tuple(mapping[x] for x in chain)
        for chain in subchains(mol)
    }


def smiles2descriptions(smiles):
    return np.array(list(map(description, map(read_smiles, smiles))))


def combine(p1, p2):
    return tuple(
        e1 if e1 == e2 else None
        for e1, e2 in zip(p1, p2)
    )


def leq(p1, p2):
    return all(
        e1 is None or e1 == e2
        for e1, e2 in zip(p1, p2)
    )


def similarity(desc1, desc2):
    common_subchains = {
        combine(p1, p2)
        for p1 in desc1
        for p2 in desc2
    }

    maximals = set()
    for p in common_subchains:
        add = True
        remove = []
        for m in maximals:
            if leq(m, p):
                remove.append(m)
            elif leq(p, m):
                add = False
        for m in remove:
            maximals.remove(m)
        if add:
            maximals.add(p)
    return maximals


class MoleculeLazyPatternClassifierProjected(LazyPatternClassifier):
    @classmethod
    def _get_pattern(cls, mol1, mol2):
        return (similarity(mol1, mol2),)

    @classmethod
    def _satisfy(cls, pattern, mols):
        result = np.empty(mols.size, dtype=bool)
        for i, mol in enumerate(mols):
            result[i] = (cls._get_pattern(pattern, mol)[0] == pattern)
        return result

    
    def fit(self, X, y):
        #Xnum, y = check_X_y(X, y)
        Xnum = X

        self.n_features_in_ = 1
        self.classes_ = unique_labels(y)
        if self.classes_.size > 2:
            raise ValueError

        y = (y == self.classes_[-1])
        self.Xnum_p_ = Xnum[y]
        self.Xnum_n_ = Xnum[~y]
        self._set_weights(Xnum, y)
        return self

    def predict(self, X):
        check_is_fitted(self, ['Xnum_p_', 'Xnum_n_'])
        #Xnum = check_array(X)
        Xnum = X

        if self.classes_.size == 1:
            return np.full(Xnum.shape[0], self.classes_[0])

        y_pred = np.empty(Xnum.shape[0], dtype=int)
        for i in range(Xnum.shape[0]):
            y_pred[i] = self._predict_one(Xnum[i])
        return self.classes_[y_pred]
