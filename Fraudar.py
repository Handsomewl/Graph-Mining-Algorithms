# From paper FRAUDAR: Bounding Graph Fraud in the Face of Camouflage
# https://dl.acm.org/doi/10.1145/2939672.2939747

import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix
from greedy import fastGreedyDecreasing


class Fraudar():
    def __init__(self, data_mat):
        self.data = data_mat

    def logWeightedAveDegree(M):
        (m, n) = M.shape
        colSums = M.sum(axis=0)
        colWeights = 1.0 / np.log(np.squeeze(colSums.A) + 5)
        colDiag = lil_matrix((n, n))
        colDiag.setdiag(colWeights)
        W = M * colDiag
        print("finished computing weight matrix")
        return fastGreedyDecreasing(W)

    def run(self, out_path="./", file_name="out", k=1, maxsize=-1):
        sparse_matrix = self.data.to_scipy()
        sparse_matrix = sparse_matrix.asfptype()
        Mcur = sparse_matrix.copy().tolil()
        res = []
        t = 0
        while t < k:
            set_row, set_col, score = logWeightedAveDegree(Mcur)
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
