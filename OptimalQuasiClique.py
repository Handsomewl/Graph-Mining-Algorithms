# From paper "Denser than the Densest Subgraph:Extracting Optimal Quasi-Cliques with Quality Guarantees"
# https://dl.acm.org/doi/10.1145/2487575.2487645

import numpy as np
import scipy.sparse.linalg as slin
from MinTree import MinTree
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


class OQC():
    # input : lil_matrix for undirected graph
    def __init__(self, data_mat, alpha=1/3):
        self.data = data_mat
        self.alpha = alpha

    def checkScore(self, G, selector):
        # G : lil_matrix
        # selector: array_like
        score = G[selector, :][:, selector].sum()
        aveScore = score / (len(selector) * (len(selector)-1))
        return aveScore

    # find top_k opqimal-quasi-clique
    def run(self, k=1, type='greedy'):
        Mcur = self.data.tolil()
        res = []
        for idx in range(k):
            if type == 'greedy':
                finalSet, score = self.greedyOQC(Mcur)
            elif type == 'naive':
                finalSet, score = self.fastGreedyDecreasing(Mcur)
            """ 
            elif type == 'ls':
                finalSet,score = self.lsOQC(Mcur)
            """
            res.append([finalSet, score])
            print(
                f'{idx} quasi-clique, size:{len(finalSet)}, score:{score},clique_density:{self.checkScore(Mcur,list(finalSet))}')
            # delete the nodes in clique already found
            for a, b in zip(Mcur.nonzero()[0], Mcur.nonzero()[1]):
                if a in finalSet or b in finalSet:
                    Mcur[a, b] = 0
        return res

    # greedy deletion to find the optimal-quasi-clique
    # metric: E(S)-alpha*|S|*(|S|-1)/2
    def greedyOQC(self, M):
        # Mcur : lil_matrix and sysmmetric matrix.
        # alpha : positive number belong to (0,1)
        Mcur = M.tolil()
        edges = Mcur.sum() / 2   # sum of edges
        Set = set(range(0, Mcur.shape[1]))
        bestScore = curScore = edges - self.alpha * (len(Set)*(len(Set)-1))/2
        deltas = np.squeeze(Mcur.sum(axis=1).A)
        tree = MinTree(deltas)
        numDeleted = 0
        deleted = []
        bestNumDeleted = 0
        while len(Set) > 2:
            node, val = tree.getMin()
            edges -= val
            # Update priority
            for j in Mcur.rows[node]:
                delt = Mcur[node, j]
                tree.changeVal(j, -delt)
            Set -= {node}
            tree.changeVal(node, float('inf'))
            deleted.append(node)
            numDeleted += 1
            curScore = edges - self.alpha * (len(Set)*(len(Set)-1))/2
            if curScore > bestScore:
                bestScore = curScore
                bestNumDeleted = numDeleted
        # reconstruct the best sets
        finalSet = set(range(0, Mcur.shape[1]))
        for idx in range(bestNumDeleted):
            finalSet.remove(deleted[idx])
        return finalSet, bestScore

    # greedy deletion to find the clique
    # metric: E(S)/(|S|*(|S|-1))
    def fastGreedyDecreasing(self, G):
        # Mcur : lil_matrix and sysmmetric matrix.
        Mcur = G.tolil()
        curScore = Mcur.sum() / 2
        Set = set(range(0, Mcur.shape[1]))
        bestAveScore = 2 * curScore / (len(Set)*(len(Set)-1))
        Deltas = np.squeeze(Mcur.sum(axis=1).A)
        tree = MinTree(Deltas)
        numDeleted = 0
        deleted = []
        bestNumDeleted = 0
        while len(Set) > 2:
            node, val = tree.getMin()
            curScore -= val
            # Update priority
            for j in Mcur.rows[node]:
                delt = Mcur[node, j]
                tree.changeVal(j, -delt)
            Set -= {node}
            tree.changeVal(node, float('inf'))
            deleted.append(node)
            numDeleted += 1
            curAveScore = 2 * curScore / (len(Set)*(len(Set)-1))
            if curAveScore > bestAveScore:
                bestAveScore = curAveScore
                bestNumDeleted = numDeleted
        # reconstruct the best sets
        finalSet = set(range(0, Mcur.shape[1]))
        for idx in range(bestNumDeleted):
            finalSet.remove(deleted[idx])
        return finalSet, bestAveScore
