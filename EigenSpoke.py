# From paper EigenSpokes: Surprising Patterns and Scalable Community Chipping in Large Graphs
# http://www.cs.cmu.edu/~christos/PUBLICATIONS/pakdd10-eigenspokes.pdf

import numpy as np
import scipy.sparse.linalg as slin
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix


class EigenSpokes():
    def __init__(self, data_mat):
        self.data = data_mat

    def run(self, k=10):
        # Truncated SVD to get the best rank-k approximation
        sparse_matrix = self.data.to_scipy()
        sparse_matrix = sparse_matrix.asfptype()
        RU, RS, RVt = slin.svds(sparse_matrix, k)
        RV = np.transpose(RVt)
        U, S, V = np.flip(RU, axis=1), np.flip(RS), np.flip(RV, axis=1)

        n_row = U.shape[0]
        n_col = V.shape[0]

        x_lower_bound = -1 / np.sqrt(n_col + 1)
        y_lower_bound = -1 / np.sqrt(n_col + 1)
        x_upper_bound = 1 / np.sqrt(n_col + 1)
        y_upper_bound = 1 / np.sqrt(n_col + 1)

        real_idx1 = S.shape[0] - 1
        real_idx2 = S.shape[0] - 2

        x = U[:, real_idx1]
        y = U[:, real_idx2]
        
        # Detect outliers by threshold/bound
        list_x_lower_outliers = [idx for idx in range(len(x)) if x[idx] > x_lower_bound]
        list_y_lower_outliers = [idx for idx in range(len(y)) if y[idx] > y_lower_bound]
        list_x_upper_outliers = [idx for idx in range(len(x)) if x[idx] < x_upper_bound]
        list_y_upper_outliers = [idx for idx in range(len(y)) if y[idx] < y_upper_bound]

        outliers_idx = list(set(list_x_lower_outliers) & set(list_y_lower_outliers) &
                            set(list_x_upper_outliers) & set(list_y_upper_outliers))
        inliers_idx = list(set(range(len(x))).difference(outliers_idx))

        outliers_idx, inliers_idx = inliers_idx, outliers_idx
        print("Outliters:",outliers_idx)

        return outliers_idx
