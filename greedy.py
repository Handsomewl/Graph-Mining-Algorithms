import numpy as np
from MinTree import MinTree
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, lil_matrix

# Greedy algorithm for find dense subgraph in bipartite graph!
def fastGreedyDecreasing(M):
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
