from sklearn import metrics
import numpy as np
import networkx as nx
import community
from munkres import Munkres


def compute_nmi(pred, labels):
    return metrics.normalized_mutual_info_score(labels, pred)

def compute_acc(pred, labels):
    return metrics.accuracy_score(labels, pred)

def computer_macrof1(pred, labels):
    return metrics.f1_score(labels, pred, average='macro')

def computer_microf1(pred, labels):
    return metrics.f1_score(labels, pred, average='micro')

def computer_ari(true_labels, pred_labels):
    return metrics.adjusted_rand_score(true_labels, pred_labels)

def cal_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
       accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def modularity(adj: np.array, pred: np.array):
    graph = nx.from_numpy_matrix(adj)
    part = pred.tolist()
    index = range(0, len(part))
    dic = zip(index, part)
    part = dict(dic)
    modur = community.modularity(part, graph)
    return modur

class clustering_metrics():
    def __init__(self, true_label):
        self.true_label = true_label
        # self.pred_label = predict_label
    def _set_pred_label(self, predict_label):
        self.pred_label = predict_label

    def clusteringAcc(self):

        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]
                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)
        idx = indexes[2][1]
        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, f1_micro

    def evaluationClusterModelFromLabel(self, epoch = None):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, f1_micro = self.clusteringAcc()
        return acc, nmi, f1_macro, ari
