# From paper FRAUDAR: Bounding Graph Fraud in the Face of Camouflage
# https://dl.acm.org/doi/10.1145/2939672.2939747

import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix
from MinTree import MinTree


class Fraudar():
    # input : lil_matrix for bipartite graph
    def __init__(self, data_mat):
        self.data = data_mat

    def logWeightedAveDegree(self, M):
        # M: scipy sparse matrix
        (m, n) = M.shape
        colSums = M.sum(axis=0)
        colWeights = 1.0 / np.log(np.squeeze(colSums.A) + 5)
        colDiag = lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        W = M * colDiag
        print("finished computing weight matrix")
        return W

    def run(self, k=1, maxsize=-1, out_path="./", file_name="out"):
        Mcur = self.data.copy().tolil()
        res = []
        t = 0
        while t < k:
            weight_matrix = self.logWeightedAveDegree(Mcur)
            set_row, set_col, score = self.fastGreedyDecreasing(weight_matrix)
            list_row, list_col = list(set_row), list(set_col)
            print("Fraudar iter %s finished." % t)

            if isinstance(maxsize, int):
                if maxsize == -1 or (maxsize >= len(list_row) and maxsize >= len(list_col)):
                    t += 1
                    res.append((list_row, list_col, score))
            elif maxsize[0] >= len(list_row) and maxsize[1] >= len(list_col):
                t += 1
                res.append((list_row, list_col, score))

            np.savetxt("%s_%s.rows" % (out_path + file_name, t),
                       np.array(list_row).reshape(-1, 1), fmt='%d')
            np.savetxt("%s_%s.cols" % (out_path + file_name, t),
                       np.array(list_col).reshape(-1, 1), fmt='%d')

            print("score obtained is ", score)

            if t >= k:
                break

            rs, cs = Mcur.nonzero()  # (u, v)
            # only delete inner connections
            rowSet = set(list_row)
            colSet = set(list_col)
            for i in range(len(rs)):
                if rs[i] in rowSet and cs[i] in colSet:
                    Mcur[rs[i], cs[i]] = 0

    def fastGreedyDecreasing(self, M):
        (m, n) = M.shape
        Ml = M.tolil()
        Mlt = M.transpose().tolil()
        rowSet = set(range(0, m))
        colSet = set(range(0, n))
        curScore = M[list(rowSet), :][:, list(colSet)].sum(axis=None)
        bestAveScore = curScore / (len(rowSet) + len(colSet))
        bestSets = (rowSet, colSet)
        print("finished setting up greedy")
        # *decrease* in total weight when *removing* this row
        # Prepare the min priority tree to begin greedy algorithm.
        rowDeltas = np.squeeze(M.sum(axis=1).A)
        colDeltas = np.squeeze(M.sum(axis=0).A)
        print("finished setting deltas")
        rowTree = MinTree(rowDeltas)
        colTree = MinTree(colDeltas)
        print("finished building min trees")

        numDeleted = 0
        deleted = []
        bestNumDeleted = 0

        while rowSet and colSet:
            if (len(colSet) + len(rowSet)) % 100000 == 0:
                print("current set size = ", len(colSet) + len(rowSet))
            nextRow, rowDelt = rowTree.getMin()
            nextCol, colDelt = colTree.getMin()
            if rowDelt <= colDelt:
                curScore -= rowDelt
                # Update priority for the node with min priority and its neighbors
                for j in Ml.rows[nextRow]:
                    delt = Ml[nextRow, j]
                    colTree.changeVal(j, -delt)
                rowSet -= {nextRow}
                rowTree.changeVal(nextRow, float('inf'))
                deleted.append((0, nextRow))
            else:
                curScore -= colDelt
                # Update priority for the node with min priority and its neighbors
                for i in Mlt.rows[nextCol]:
                    delt = Ml[i, nextCol]
                    rowTree.changeVal(i, -delt)
                colSet -= {nextCol}
                colTree.changeVal(nextCol, float('inf'))
                deleted.append((1, nextCol))

            numDeleted += 1
            curAveScore = curScore / (len(colSet) + len(rowSet))
            if curAveScore > bestAveScore:
                bestAveScore = curAveScore
                bestNumDeleted = numDeleted

        # Reconstruct the best row and column sets
        finalRowSet = set(range(m))
        finalColSet = set(range(n))
        for i in range(bestNumDeleted):
            if deleted[i][0] == 0:
                finalRowSet.remove(deleted[i][1])
            else:
                finalColSet.remove(deleted[i][1])
        return (finalRowSet, finalColSet, bestAveScore)
