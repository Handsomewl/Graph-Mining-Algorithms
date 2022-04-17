import copy
from pyexpat import model
import numpy as np
from MinTree import MinTree
from collections import defaultdict
from scipy.sparse import lil_matrix, csc_matrix, csr_matrix, coo_matrix


class FlowScope():
    '''
    Parameters
    ----------
    graphList: list
        Graph instance contains adjency matrix, and possible multiple signals.
        list of lil_matrix !!!
    '''

    def __init__(self, graphList: list):
        self.k = 1
        self.outpath = ''
        self.AB, self.B, self.BC = graphList

    def run(self):
        self.nres = []
        self.initData()
        for i in range(self.k):
            finalsets, score = self.fastGreedyDecreasing()
            self.nres.append(finalsets)
        return self.nres

    def initData(self):
        print(self.AB.shape, self.B.shape, self.BC.shape)
        self.sets_ori = []
        self.sets_ori.append(set(range(self.AB.shape[0])))
        self.sets_ori.append(set(range(self.B.shape[0])))
        self.sets_ori.append(set(range(self.BC.shape[0])))
        print(len(self.sets_ori[0]), len(
            self.sets_ori[1]), len(self.sets_ori[2]))

    # find the smallest one in all set
    def findmin(self):
        min_tree_i = -1
        min_indices = -1
        min_value = float('inf')
        for i in range(len(self.dtrees)):
            index, value = self.dtrees[i].getMin()
            if value < min_value:
                min_value = value
                min_tree_i = i
                min_indices = index
        return min_tree_i, min_indices, value

    def checkset(self, sets):
        res = True
        for i in range(len(sets)):
            if len(sets[i]) == 0:
                res = False
                break
        return res

    def initGreedy(self):
        self.sets = []
        self.dtrees = []
        self.curScore = 0
        self.bestAveScore = 0
        self.bestNumDeleted = defaultdict(int)
        self.deltas1 = np.array(np.squeeze(self.AB.sum(axis=1).A))
        self.deltas2 = np.array(np.squeeze(self.AB.sum(axis=0).A)) + np.array(
            np.squeeze(self.BC.sum(axis=1).A))+np.array(np.squeeze(self.B.sum(axis=0).A))
        self.deltas3 = np.array(np.squeeze(self.BC.sum(axis=0).A))
        self.dtrees = [MinTree(self.deltas1), MinTree(
            self.deltas2), MinTree(self.deltas3)]
        # 因为每一条边被计算了两次
        self.curScore = (sum(self.deltas1) +
                         sum(self.deltas2) + sum(self.deltas3)) / 2
        self.sets.append(set(range(self.AB.shape[0])))
        self.sets.append(set(range(self.B.shape[0])))
        self.sets.append(set(range(self.BC.shape[0])))
        s = sum([len(self.sets[i]) for i in range(len(self.sets))])
        print('s size: ', s)
        curAveScore = self.curScore / s
        print('initial score of g: ', curAveScore)
        self.bestAveScore = curAveScore

    def updateConnNode(self, min_tree_i, idx, val):
        if min_tree_i == 0:    # delete a node from A
            for j in self.AB.rows[idx]:
                self.dtrees[1].changeVal(j, -self.AB[idx, j])
        elif min_tree_i == 2:  # delete a node from C
            for j in self.BC.T.rows[idx]:
                self.dtrees[1].changeVal(j, -self.BC[j, idx])
        elif min_tree_i == 1:  # delete a node from B
            for j in self.AB.T.rows[idx]:
                self.dtrees[0].changeVal(j, -self.AB[j, idx])
            for j in self.BC.rows[idx]:
                self.dtrees[2].changeVal(j, -self.BC[idx, j])
            for j in self.B.rows[idx]:
                self.dtrees[1].changeVal(j, -self.B[idx, j])

        self.curScore -= val
        self.sets[min_tree_i] -= {idx}
        self.dtrees[min_tree_i].changeVal(idx, float('inf'))
        self.deleted[min_tree_i].append(idx)
        self.numDeleted[min_tree_i] += 1

    def fastGreedyDecreasing(self):
        print('This is the cpu version of FlowScope')
        print('Start  greedy')

        self.initGreedy()
        self.numDeleted = defaultdict(int)
        self.deleted = {}
        for i in range(len(self.sets)):
            self.deleted[i] = []
        finalsets = copy.deepcopy(self.sets)

        # repeat deleting until one node set is empty
        while self.checkset(self.sets):
            min_tree_i, idx, val = self.findmin()
            self.updateConnNode(min_tree_i, idx, val)
            s = sum([len(self.sets[i]) for i in range(len(self.sets))])
            curAveScore = self.curScore / s
            if curAveScore >= self.bestAveScore and self.checkset(self.sets):
                for i in range(len(self.sets)):
                    self.bestNumDeleted[i] = self.numDeleted[i]
                self.bestAveScore = curAveScore

        print('best delete number : ', self.bestNumDeleted)
        print('nodes number remaining:  ', len(
            self.sets[0]), len(self.sets[1]), len(self.sets[2]))
        print(f'best score of g(S): {self.bestAveScore},')

        for i in range(len(finalsets)):
            best_deleted = self.deleted[i][:self.bestNumDeleted[i]]
            finalsets[i] = finalsets[i] - set(best_deleted)
        print('size of found subgraph:  ', len(
            finalsets[0]), len(finalsets[1]), len(finalsets[2]))
        return finalsets, self.bestAveScore
