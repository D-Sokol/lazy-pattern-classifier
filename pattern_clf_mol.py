import itertools
import networkx as nx
import networkx.algorithms.isomorphism as nxiso
import numpy as np
from pysmiles import read_smiles

from pattern_clf import LazyPatternClassifier

# Too slow
def intersection(mol1, mol2):
    G = nx.Graph()
    for ix1, el1 in mol1.nodes(data='element'):
        for ix2, el2 in mol2.nodes(data='element'):
            if el1 == el2:
                G.add_node((ix1, ix2))
    for (i1, i2), (j1, j2) in itertools.combinations(G.nodes, r=2):
        if i1 != j1 and i2 != j2 and mol1.has_edge(i1, j1) == mol2.has_edge(i2, j2):
            G.add_edge((i1,i2), (j1,j2))

    cliques = list(nx.find_cliques(G))
    maxlen = max(map(len, cliques))
    cliques = [c for c in cliques if len(c) == maxlen]
    subgraphs = [mol1.subgraph([p[0] for p in c]) for c in cliques]
    subgraphs = [sg for sg in subgraphs if nx.is_connected(sg)]
    return subgraphs


_cmp = nxiso.categorical_node_match('element', None)
def satisfy(pattern, mol):
    return all(
        nxiso.GraphMatcher(mol, subgraph, node_match=_cmp).subgraph_is_isomorphic()
        for subgraph in pattern
    )


def to_smiles_array(strings: np.ndarray) -> np.ndarray:
    mols = list(map(read_smiles, strings))
    return np.array(mols + [None], dtype=object)[:-1]
    

class MoleculeLazyPatternClassifier(LazyPatternClassifier):
    @classmethod
    def _get_pattern(cls, mol1, mol2):
        return (intersection(mol1, mol2),)

    @classmethod
    def _satisfy(cls, pattern, mols):
        result = np.empty(mols.size, dtype=bool)
        for i, mol in enumerate(mols):
            pass
        return result
