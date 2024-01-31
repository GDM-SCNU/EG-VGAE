# coding=utf-8
# Author: Jung
# Time: 2023/3/12 15:06

import utils
import evaluate
import torch
import numpy as np
import dgl
import pickle  as pkl
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import scipy.sparse as sp
import networkx as nx
import os
import random
from dgl.nn import GraphConv
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import math
random.seed(826)
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)

DATASET = "pubmed"

def load_data(name='cora'):
    root = "D:\PyCharm_WORK\MyCode\Jung\datasets\\"
    dataset_addr = root + name + ".pkl"
    with open(dataset_addr, "rb")as f:
        dataset = pkl.load(f)
    adj = dataset['adj']
    feat = dataset['feat']
    label = dataset['label']

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = utils.sparse_to_tuple(adj_label)
    adj_norm = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = utils.sparse_to_tuple(adj_norm)
    feat = utils.sparse_to_tuple(feat)

    adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                        torch.Size(adj_norm[2]))
    adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                         torch.Size(adj_label[2]))
    feat = torch.sparse.FloatTensor(torch.LongTensor(feat[0].T), torch.FloatTensor(feat[1]),
                                        torch.Size(feat[2]))

    weight_mask_orig = adj_label.to_dense().view(-1) == 1
    weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
    weight_tensor_orig[weight_mask_orig] = pos_weight_orig


    return adj, adj_norm, adj_label, feat, label, norm, weight_tensor_orig


def create_kcore_graph(sp_adj):
    nx_graph = nx.from_scipy_sparse_matrix(sp_adj)
    full_node_list = nx.nodes(nx_graph)
    core_num_dict = nx.core_number(nx_graph)
    print("unique core nums: ", len(np.unique(np.array(list(core_num_dict.values())))))
    max_core_num = max(list(core_num_dict.values()))
    print('max core num: ', max_core_num)

    def get_format_str(cnt):
        max_bit = 0
        while cnt > 0:
            cnt //= 10
            max_bit += 1
        format_str = '{:0>' + str(max_bit) + 'd}'
        return format_str

    format_str = get_format_str(max_core_num)


    for i in range(1, max_core_num + 1):
        k_core_graph = nx.k_core(nx_graph, k = i, core_number=core_num_dict)
        k_core_graph.add_nodes_from(full_node_list)
        A = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=full_node_list)



        signature = format_str.format(i)
        sp.save_npz(os.path.join("datasets\\"+DATASET+"\kcore_graph", signature + '.npz'), A)


def get_kcore_graph():
    date_dir_list = sorted(os.listdir("datasets\\"+ DATASET + "\kcore_graph"))
    norm_list = []
    weight_list = []
    adj_norm_list = []
    adj_recons_list = []
    adj_for_w_list = []
    index_list = []
    for i, file_name in enumerate(date_dir_list):
        adj = sp.load_npz(os.path.join("datasets\\"+ DATASET + "\kcore_graph", file_name)) # 无自环
        adj_for_w_list.append(adj)
        index_list.append(np.unique(adj.tocoo().row))

        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = utils.sparse_to_tuple(adj_label)
        adj_norm = utils.normalize_adj(adj + sp.eye(adj.shape[0]))
        adj_norm = utils.sparse_to_tuple(adj_norm)
        adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]),
                                            torch.Size(adj_norm[2]))
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]),
                                             torch.Size(adj_label[2]))
        weight_mask_orig = adj_label.to_dense().view(-1) == 1
        weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
        weight_tensor_orig[weight_mask_orig] = pos_weight_orig

        norm_list.append(norm)
        weight_list.append(weight_tensor_orig)
        adj_norm_list.append(adj_norm)
        adj_recons_list.append(adj_label)

    return adj_for_w_list, adj_norm_list, adj_recons_list, norm_list, weight_list, index_list





class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = self.__random_uniform_init(input_dim, output_dim)
        self.activation = activation

    def __random_uniform_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
        return nn.Parameter(initial)

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class KMVGAE(nn.Module):
    def __init__(self, **kwargs):
        super(KMVGAE, self).__init__()
        self.adj = kwargs['adj']
        self.feat = kwargs['feat']
        self.label = kwargs['label']
        self.norm = kwargs['norm']
        self.weight_tensor_orig = kwargs['weight_tensor_orig']
        self.adj_loop = kwargs['adj_loop']
        self.num_node, self.feat_dim = self.feat.shape
        self.nClusters = len(np.unique(self.label))


        # read k-core data
        self.k_core_adj = kwargs['adj_norm_list']
        self.k_core_orgadj = kwargs['adj_recons_list']
        self.k_core_norm = kwargs['norm_list']
        self.k_core_weight = kwargs['weight_list']
        self.adj_for_w = kwargs['adj_for_w_list']
        self.k_core_index = kwargs['index_list']


        hid1_dim = 32 # #32
        self.hid2_dim = 16 # 16

        # VGAE training parameters
        self.activation = F.relu

        self.base_gcn = GraphConvSparse(self.feat_dim, hid1_dim, self.activation)
        self.gcn_mean = GraphConvSparse(hid1_dim, self.hid2_dim, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hid1_dim, self.hid2_dim, activation = lambda x:x)

        # GMM training parameters
        self.pi = nn.Parameter(torch.ones(self.nClusters) / self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.hid2_dim), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.hid2_dim), requires_grad=True)


    def _encode(self, adj, i):
        hidden = self.base_gcn(self.feat, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(self.num_node, self.hid2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z

    @staticmethod
    def _decode(z):
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred


    def _get_w(self, pred, i, max_nmi):


        ratio = 1 - max_nmi
        drop = np.random.choice(self.k_core_index[i], size=(int(len(self.k_core_index[i]) * ratio)))
        p = torch.zeros((self.num_node, self.nClusters))
        p[range(self.num_node), pred] = 1
        W = (p @ p.t()).numpy().astype(np.float)
        W = W * adj_for_w_list[i].toarray()



        W = W.astype(np.float)
        W = sp.csr_matrix(W)
        W = utils.sparse_to_tuple(W)
        W = torch.sparse.FloatTensor(torch.LongTensor(W[0].T), torch.FloatTensor(W[1]),
                                        torch.Size(W[2])) # 2708 * 2708
        W = W.unsqueeze(-1) # 2708, 2708, 1
        self.W_tile = torch.cat([W] * self.nClusters, - 1)


    def _pretrain(self, i):
        if not os.path.exists('datasets/'+ DATASET + '/pretrain/model_'+str(0)+'.pk'):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01) # lr = 0.01 for cora citeseer & pubmed
            epoch_bar = tqdm(range(200))
            nmi_best = 0
            gmm = GaussianMixture(n_components=self.nClusters, covariance_type='diag')
            nmi_list = []
            for _ in epoch_bar:
                optimizer.zero_grad()
                _, _, z = self._encode(self.k_core_adj[i], i)
                A_pred = self._decode(z)
                loss = self.k_core_norm[i] * F.binary_cross_entropy(A_pred.view(-1), self.k_core_orgadj[i].to_dense().view(-1), weight = self.k_core_weight[i])
                loss.backward()
                optimizer.step()
                y_pred = gmm.fit_predict(z.detach().numpy())
                self.pi.data = torch.from_numpy(gmm.weights_).to(torch.float)
                self.mu_c.data = torch.from_numpy(gmm.means_).to(torch.float)
                self.log_sigma2_c.data =  torch.log(torch.from_numpy(gmm.covariances_)).to(torch.float)
                acc = cal_acc(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])
                nmi = evaluate.compute_nmi(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                f1 = evaluate.computer_macrof1(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                ari = evaluate.computer_ari(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
                epoch_bar.write('Loss pretraining = {:.4f}, acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}'.format(loss, acc , nmi, f1, ari))
                nmi_list.append(acc)
                if (nmi > nmi_best):
                  nmi_best = nmi
                  self.logstd = self.mean
                  torch.save(self.state_dict(), 'datasets/'+ DATASET + '/pretrain/model_'+str(0)+'.pk')
            print("Best accuracy : ",nmi_best)
        else:
            self.load_state_dict(torch.load('datasets/'+ DATASET + '/pretrain/model_'+str(0)+'.pk'))

    def _train(self, i, constrained):
        self.load_state_dict(torch.load('datasets/'+ DATASET + '/pretrain/model_'+str(0)+'.pk'))
        optimizer = torch.optim.Adam(self.parameters(), lr=0.002)
        acc_list = []
        nmi_list = []
        max_nmi = 0
        epoch_bar = tqdm(range(200))
        for epoch in epoch_bar:
            optimizer.zero_grad()
            z_mean, z_logstd, z = self._encode(self.k_core_adj[i], i)
            A_pred = self._decode(z)

            loss = self._ELBO(A_pred, z, z_mean, z_logstd, i, constrained = constrained)


            y_pred, prob_y_pred = self._predict(z)
            acc = cal_acc(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])
            nmi = evaluate.compute_nmi(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            f1 = evaluate.computer_macrof1(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            ari = evaluate.computer_ari(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])


            q_adj = self.adj_for_w[i]
            q_adj = q_adj[self.k_core_index[i],:]
            q_adj = q_adj[:, self.k_core_index[i]]
            q = evaluate.modularity(q_adj.toarray(), y_pred[self.k_core_index[i]])
            epoch_bar.write(
                'Loss training = {:.4f}, acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}, q = {:.4f}'.format(loss, acc, nmi, f1, ari, q))
            acc_list.append(acc)
            nmi_list.append(nmi)

            if nmi > max_nmi:
                max_nmi = nmi
                best_y = y_pred

            loss.backward()
            optimizer.step()

        print("MAX ACC : {}, MAX NMI :{}".format(max(acc_list), max(nmi_list)))
        self._get_w(best_y, i, max_nmi)

    def _ELBO_No_W(self, A_pred, z, z_mean, z_logstd):
        """
            消融实验使用的损失函数，没有约束信息
        """
        pi = self.pi
        mean_c = self.mu_c
        logstd_c = self.log_sigma2_c
        det = 1e-2

        loss_recons = 1e-2 * norm * F.binary_cross_entropy(A_pred.view(-1), self.adj_loop.to_dense().view(-1), weight= weight_tensor_orig)
        loss_recons = loss_recons * self.num_node

        gamma_c = torch.exp(torch.log(pi.unsqueeze(0)) + self._gaussian_pdfs_log(z, mean_c, logstd_c)) + det
        gamma_c = gamma_c / (gamma_c.sum(1).view(-1,1))

        KL1 = 0.5 * torch.mean(torch.sum(gamma_c * torch.sum(logstd_c.unsqueeze(0) + torch.exp(z_logstd.unsqueeze(1) - logstd_c.unsqueeze(0))
                                                             + (z_mean.unsqueeze(1) - mean_c.unsqueeze(0)).pow(2) / torch.exp(logstd_c.unsqueeze(0)), 2), 1))
        KL2 = torch.mean(torch.sum(gamma_c * torch.log(pi.unsqueeze(0) / (gamma_c)),1)) + 0.5 * torch.mean(torch.sum(1 + z_logstd, 1))
        loss_clus = KL1 - KL2

        loss_elbo = loss_recons + loss_clus
        return loss_elbo


    def _ELBO(self, A_pred, z, z_mean, z_logstd, i, constrained = False):
        pi = self.pi
        mean_c = self.mu_c
        logstd_c = self.log_sigma2_c
        det = 1e-2

        if constrained is True:
            w = self.W_tile.to_dense()  # 2707， 2708
            y_pred = torch.exp(torch.log(pi.unsqueeze(0)) + self._gaussian_pdfs_log(z, mean_c, logstd_c))
            mul = w.__mul__(y_pred)
            sum_j = torch.sum(mul, dim=-2) # + det

        loss_recons = 1e-2 * self.k_core_norm[i] * F.binary_cross_entropy(A_pred.view(-1), self.k_core_orgadj[i].to_dense().view(-1), weight= self.k_core_weight[i])
        loss_recons = loss_recons * self.num_node

        gamma_c = torch.exp(torch.log(pi.unsqueeze(0)) + self._gaussian_pdfs_log(z, mean_c, logstd_c)) + det # 2708,7
        gamma_c = gamma_c / (gamma_c.sum(1).view(-1,1)) # 2708,7


        KL1 = 0.5 * torch.mean(torch.sum(gamma_c * torch.sum(logstd_c.unsqueeze(0) + torch.exp(z_logstd.unsqueeze(1) - logstd_c.unsqueeze(0))
                                                             + (z_mean.unsqueeze(1) - mean_c.unsqueeze(0)).pow(2) / torch.exp(logstd_c.unsqueeze(0)), 2), 1))

        if constrained is True:
            KL2 = torch.mean(torch.sum(gamma_c * torch.log((sum_j / gamma_c) + det), 1)) + 0.5 * torch.mean(torch.sum(1 + z_logstd, 1))
        else:
            KL2 = torch.mean(torch.sum(gamma_c * torch.log(pi.unsqueeze(0) / (gamma_c)), 1)) + 0.5 * torch.mean(torch.sum(1 + z_logstd, 1))

        loss_clus = KL1 - KL2
        loss_elbo = loss_recons + loss_clus
        return loss_elbo

    def _gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self._gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)


    def _gaussian_pdf_log(self,x,mu,log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def _predict(self, z):
        pi = self.pi
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        det = 1e-2
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self._gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita = yita_c.detach().numpy()
        return np.argmax(yita, axis=1), yita_c.detach()

    def _downstream(self):

        snpa = 0

        self.load_state_dict(torch.load('datasets/' + DATASET + '/evoluation/evo_'+str(snpa)+'.pk'))
        i = snpa
        with torch.no_grad():
            _, _, z = self._encode(self.k_core_adj[i], i)
            y_pred, _ = self._predict(z)
            acc = cal_acc(self.label[self.k_core_index[i]], y_pred[self.k_core_index[i]])
            nmi = evaluate.compute_nmi(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            f1 = evaluate.computer_f1(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            ari = evaluate.computer_ari(y_pred[self.k_core_index[i]], self.label[self.k_core_index[i]])
            q_adj = self.adj_for_w[i]
            q_adj = q_adj[self.k_core_index[i], :]
            q_adj = q_adj[:, self.k_core_index[i]]
            q = utils.modularity(q_adj.toarray(), y_pred[self.k_core_index[i]])
            print("downsteam := acc = {:.4f}, nmi = {:.4f}, f1 = {:.4f}, ari = {:.4f}, q = {:.4f}".format(acc,nmi,f1,ari,q))

        kwargs = {
            "early_exaggeration": 100, # 38
            "init": "pca",
            "perplexity": 100,
            "emb": z,
            "label": self.label,
            "path": 'datasets/' + DATASET + '/evoluation/evo_'+str(snpa)+'.pdf'
        }

        utils.visualization(**kwargs)


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

if __name__ == "__main__":
    adj, adj_norm, adj_label, feat, label, norm, weight_tensor_orig = load_data(DATASET)
    # create_kcore_graph(adj)
    adj_for_w_list, adj_norm_list, adj_recons_list, norm_list, weight_list, index_list = get_kcore_graph()

    num_k_core = len(adj_norm_list)

    configs = {
        "adj": adj_norm,
        "feat": feat,
        "label": label,
        "norm": norm,
        "weight_tensor_orig" : weight_tensor_orig,
        "adj_loop" : adj_label,
        "adj_norm_list" : adj_norm_list,
        "adj_recons_list" : adj_recons_list,
        "norm_list" : norm_list,
        "weight_list" : weight_list,
        "adj_for_w_list": adj_for_w_list,
        "index_list": index_list
    }
    model = KMVGAE(**configs)
    model._pretrain(0)
    for i in range(num_k_core-1, -1, -1):
        print("##########       {}         ##########".format(i))

        if i == num_k_core-1:
            model._train(i, False) #
        else:
            # model._pretrain(i)
            model._train(i, False) # EGC-VGAE use True
    # model._downstream()
