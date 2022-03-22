import time
import pickle
import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.io as scio
import networkx as nx
import matplotlib.pyplot as plt

from math import sqrt
from MinTree import MinTree
from numpy import matmul
from sklearn.utils.extmath import randomized_svd, squared_norm
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix

# Norm for both dense and sparse matrix
def norm(x):
    if sp.issparse(x):
        return sp.linalg.norm(x)
    return np.linalg.norm(x)

# Create a random matrix with size and probability
def random_matrix(size, prob):
    a = np.random.random(size)
    return np.where(a<prob,1,0)

# the project function used in PGD
def projectetBack(v, lowb, upperb):
	tmp1 = np.where(v > upperb, upperb, v)
	tmp2 = np.where(tmp1 < lowb, lowb, tmp1)
	return tmp2

# Error for model the two coupled matrix
def error(G1, G2, C, U, V, W, alpha):
    return squared_norm(G1-U) + squared_norm(G2-V) + alpha * squared_norm(C-W)

# Calculate he gradient of U


def gradient_U(U, V, G1, C, alpha):
    nabla_f_u = matmul(matmul(U, V.T)-C, V)
    nabla_g_u = 2 * matmul(matmul(U, U.T)-G1, U)
    return alpha * nabla_f_u + nabla_g_u

# Calculate the gradient of Vt


def gradient_V(U, V, G2, C, alpha):
    nabla_f_v = matmul(matmul(U.T, U)-C.T, U)
    nabla_g_v = 2 * matmul(matmul(V.T, V)-G2, V)
    return alpha * nabla_f_v + nabla_g_v

# Update U,V via Adam Optimizer


def AdamUpdate(G1, G2, C, R, epochs, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, weight_decay=0):

    alpha = (norm(G1)+norm(G2)) / (2*norm(C))
    U = np.random.rand(G1.shape[0], R)
    V = np.random.rand(G2.shape[0], R)
    total_error = [error(G1, G2, C, matmul(
        U, U.T), matmul(V, V.T), matmul(U, V.T), alpha)]
    mt = vt = vt_max = np.zeros(U.shape, dtype=np.float64)
    pt = qt = qt_max = np.zeros(V.shape, dtype=np.float64)
    for t in range(1, epochs+1):
        # Update U
        gt = gradient_U(U, V, G1, C, alpha)
        # gt = gt / norm(gt) # normalization
        gt[np.isnan(gt)] = 0.01
        mt = betas[0] * mt + (1-betas[0]) * gt
        vt = betas[1] * vt + (1-betas[1]) * np.power(gt, 2)
        mt_hat = mt / (1-betas[0] ** t)
        vt_hat = vt / (1-betas[1] ** t)
        vt_max = np.maximum(vt_hat, vt_max)
        if amsgrad:
            U -= weight_decay*U + lr * np.divide(mt_hat, np.sqrt(vt_max)+eps)
        else:
            U -= weight_decay*U + lr * np.divide(mt_hat, np.sqrt(vt_hat)+eps)
        # Update V
        ht = gradient_V(U, V, G2, C, alpha)
        # ht = ht / norm(ht) # normalization
        ht[np.isnan(ht)] = 0.01
        pt = betas[0] * pt + (1-betas[0]) * ht
        qt = betas[1] * qt + (1-betas[1]) * np.power(ht, 2)
        pt_hat = pt / (1-betas[0] ** t)
        qt_hat = qt / (1-betas[1] ** t)
        qt_max = np.maximum(qt_hat, qt_max)
        if amsgrad:
            V -= weight_decay * V + lr * \
                np.divide(pt_hat, np.sqrt(qt_max) + eps)
        else:
            V -= weight_decay * V + lr * \
                np.divide(pt_hat, np.sqrt(qt_hat) + eps)
        # Calculate the reconstruction error
        total_error.append(error(G1, G2, C, matmul(U, U.T),
                           matmul(V, V.T), matmul(U, V.T), alpha))
    return U, V, total_error

# Gradient descent update


def GradientUpdate(G1, G2, C, R, epochs, lr=1e-4, weight_decay=0.01):
    alpha = (norm(G1)+norm(G2)) / (2*norm(C))
    U = np.random.rand(C.shape[0], R)
    V = np.random.rand(C.shape[1], R)
    total_error = [error(G1, G2, C, matmul(
        U, U.T), matmul(V, V.T), matmul(U, V.T), alpha)]
    for iter in range(epochs):
        # Update U
        nabla_u = gradient_U(U, V, G1, C, alpha)
        nabla_u = nabla_u / norm(nabla_u)
        nabla_u[np.isnan(nabla_u)] = 0.01
        U -= weight_decay * U + lr * nabla_u
        # Update V
        nabla_v = gradient_V(U, V, G2, C, alpha)
        nabla_v = nabla_v / norm(nabla_v)
        nabla_v[np.isnan(nabla_v)] = 0.01
        V -= weight_decay * V + lr * nabla_v
        # Calculate the reconstruction error
        total_error.append(error(G1, G2, C, matmul(U, U.T),
                           matmul(V, V.T), matmul(U, V.T), alpha))
    return U, V, total_error


# Multiplicative update of coupled matrix factorization
# Minimize the objective Fro(G1-u@u.T) + Fro(G2-v@v.T) + alpha*Fro(C-u@v.T)
# where Fro means the Frobenius norm of the matrix
def MU(G1, G2, C, R, epochs):
    # G1,G2 : lil_matrix which represent two layers
    # C : the cross-layer dependency matrix

    # Initialize U,V,alpha
    alpha = (norm(G1)+norm(G2)) / (2*norm(C))
    u = np.random.rand(G1.shape[0], R)
    v = np.random.rand(G2.shape[1], R)

    # Error of initialization
    total_error = [error(G1, G2, C, matmul(
        u, u.T), matmul(v, v.T), matmul(u, v.T), alpha)]

    for it in range(epochs):
        # Update U
        u_upper = 2 * G1 * u + alpha * matmul(C, v)
        u_lower = matmul(u, 2 * matmul(u.T, u) + alpha * matmul(v.T, v))
        u_res = np.power(np.divide(u_upper, u_lower), 1/2)
        u_res[np.isnan(u_res)] = 0.01
        u = np.multiply(u, u_res)

        # Update V
        v_upper = 2 * G2 * v + alpha * matmul(C.T, u)
        v_lower = matmul(v, 2 * matmul(v.T, v) + alpha * matmul(u.T, u))
        v_res = np.power(np.divide(v_upper, v_lower), 1/2)
        v_res[np.isnan(v_res)] = 0.01
        v = np.multiply(v, v_res)

        # Calculate the reconstruction error
        total_error.append(error(G1, G2, C, matmul(u, u.T),
                           matmul(v, v.T), matmul(u, v.T), alpha))
    return u, v, total_error


#  Multi-multiplicative updates of coupled matrix factorization
def MMU(G1, G2, G3, C12, C13, C23, R, epochs):
    alpha = (norm(G1)+norm(G2))/(2*norm(C12))
    beta = (norm(G1)+norm(G3))/(2*norm(C13))
    gamma = (norm(G2)+norm(G3))/(2*norm(C23))

    U = np.random.rand(G1.shape[0], R)
    V = np.random.rand(G2.shape[0], R)
    W = np.random.rand(G3.shape[0], R)
    # Error of initialization
    total_error = [squared_norm(G1-U@U.T)+squared_norm(G2-V@V.T)+squared_norm(G3-W@W.T) +
                   alpha*squared_norm(C12-U@V.T)+beta*squared_norm(C13-U@W.T)+gamma*squared_norm(C23-V@W.T)]

    for iter in range(epochs):
        # Update U
        u_upper = 2 * G1 @ U + alpha*C12@V + beta*C13@W
        u_lower = U @ (2*U.T@U + alpha*V.T@V + beta*W.T@W)
        u_res = np.power(np.divide(u_upper, u_lower), 1/2)
        u_res[np.isnan(u_res)] = 0.01
        U = np.multiply(U, u_res)
        # Update V
        v_upper = 2*G2@V + alpha*C12.T@U + gamma*C23@W
        v_lower = V @ (2*V.T@V + alpha*U.T@U + gamma*W.T@W)
        v_res = np.power(np.divide(v_upper, v_lower), 1/2)
        v_res[np.isnan(v_res)] = 0.01
        V = np.multiply(V, v_res)
        # Update W
        w_upper = 2*G3@W + beta*C13.T@U + gamma*C23.T@V
        w_lower = W @ (2*W.T@W + beta*U.T@U + gamma*V.T@V)
        w_res = np.power(np.divide(w_upper, w_lower), 1/2)
        w_res[np.isnan(w_res)] = 0.01
        W = np.multiply(W, w_res)
        # Calculate the reconstruction error
        total_error.append(squared_norm(G1-U@U.T)+squared_norm(G2-V@V.T)+squared_norm(G3-W@W.T) +
                           alpha*squared_norm(C12-U@V.T)+beta*squared_norm(C13-U@W.T)+gamma*squared_norm(C23-V@W.T))
    return U, V, W, total_error


def Greedy(G, U):
    Score = []
    selector = []
    for idx in range(U.shape[1]):
        bestSet = set()
        delta = np.sqrt(squared_norm(U[:, idx]) / U.shape[0])
        candidateSet = np.argwhere(np.array(U[:, idx]) > delta)[:, 0]
        # print("Length of candidate set", len(candidateSet))
        g = G.copy()
        g = g[candidateSet, :][:, candidateSet]
        degree = np.squeeze(g.sum(axis=0).A)          # d: degree array
        # Initialize the set to start remove greedily
        curSet = set(range(len(candidateSet)))
        tree = MinTree(degree)
        curScore = sum(degree) / 2
        bestAveScore = 2 * curScore / (len(curSet)*(len(curSet)-1))

        while len(curSet) > 2:
            node, val = tree.getMin()
            # node 是tree输入数组中最小值元素的索引
            curSet -= {node}
            curScore -= val
            # print('max_density',max_density,'node',node,'len s',len(curSet))
            tree.setVal(node, float('inf'))
            # Update priority of neighbors
            for j in g.rows[node]:
                delt = g[node, j]
                tree.changeVal(j, -delt)
            g[node, :], g[:, node] = 0, 0
            curAveScore = 2 * curScore / (len(curSet)*(len(curSet)-1))
            if curAveScore > bestAveScore:
                bestAveScore = curAveScore
                bestSet = curSet.copy()
        pos = list(bestSet)
        res = list(np.array(candidateSet)[pos])
        Score.append(bestAveScore)
        selector.append(res)
        print("Maximum density:", bestAveScore,
              "len of optimal set:", len(bestSet))
    return Score, selector


def fastGreedyDecreasing(G):
    # Mcur is a sysmmetric matrix.
    # Mcur : lil_matrix
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


def greedy(G, U):
    # G : the adjacency matrix
    # U : the factor matrix
    bestScore = []
    optRes = []
    for idx in range(U.shape[1]):
        # delta = np.sqrt(norm(U[:, idx])**2 / U.shape[0])
        delta = np.sqrt(sum(U[:, idx])**2 / U.shape[0])
        Set = list(np.argwhere(np.array(U[:, idx]) > delta)[:, 0])
        print('length of candidate set:',len(Set))
        g = G.copy().asfptype()
        g = g[Set, :][:, Set]
        
        if len(Set) > 1:
            finalSet, score = fastGreedyDecreasing(g)
            print('length of bestSet', len(finalSet), 'bestScore', score)
            pos = list(finalSet)
            res = list(np.array(Set)[pos])
            bestScore.append(score)
            optRes.append(res)
    return optRes, bestScore

# Give a graph and its indicator vector ,calculate the score.


def checkScore(G, selector):
    # G : lil_matrix
    # selector: array_like
    score = G[selector, :][:, selector].sum()
    aveScore = score / (len(selector) * (len(selector)-1))
    return aveScore

# Give a graph and its indicator vector, extend the subgraph with higher score.


def extend(g, sub):
    # g:lil_matrix
    # sub: which means subgraph that to be extend , array_like data
    # curScore : number of edges of sum of degrees
    subgraph = sub.copy()
    curScore = g[subgraph, :][:, subgraph].sum() / 2
    curAveScore = 2 * curScore / (len(subgraph)*(len(subgraph)-1))
    curSet = set(subgraph)
    # the nodes added to the subgraph
    nodes = []
    addSet = set()
    # We just simply check the neighbors of the nodes that not in the current subgraph
    # Or we can get the top_k neighbors with high degrees as the candidate set.
    for node in subgraph:
        subSet = set(g.rows[node]) - curSet
        addSet = addSet | subSet
    for neigh in addSet:
        addedges = g[neigh][:, subgraph].sum()
        tmpAveScore = 2 * (curScore+addedges) / (len(curSet)*(len(curSet)+1))
        if tmpAveScore >= curAveScore:
            curScore += addedges
            curAveScore = tmpAveScore
            curSet.add(neigh)
            subgraph.append(neigh)
            nodes.append(neigh)
    if len(nodes):
        print("Add nodes:", nodes,'length of current subgraph:',len(subgraph) ,"curAveScore:", curAveScore)
    else:
        print("No nodes add to the current subgraph!")
    return subgraph, curAveScore
