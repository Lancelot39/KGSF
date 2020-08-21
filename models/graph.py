import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.gat_conv import GATConv


def kaiming_reset_parameters(linear_module):
    nn.init.kaiming_uniform_(linear_module.weight, a=math.sqrt(5))
    if linear_module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear_module.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear_module.bias, -bound, bound)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)

        kaiming_reset_parameters(self)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, ninp, nhid, dropout=0.5):
        super(GCN, self).__init__()

        # self.gc1 = GraphConvolution(ninp, nhid)
        self.gc2 = GraphConvolution(ninp, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        """x: shape (|V|, |D|); adj: shape(|V|, |V|)"""
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
        # return F.log_softmax(x, dim=1)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        # self.a = nn.Parameter(torch.zeros(size=(2*self.dim, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        N = h.shape[0]
        assert self.dim == h.shape[1]
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # attention = F.softmax(e, dim=1)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(attention, h)

class SelfAttentionLayer_batch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer_batch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        # self.a = nn.Parameter(torch.zeros(size=(2*self.dim, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, mask):
        N = h.shape[0]
        assert self.dim == h.shape[2]
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # attention = F.softmax(e, dim=1)
        mask=1e-30*mask.float()

        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
        #print(e.size())
        #print(mask.size())
        attention = F.softmax(e+mask.unsqueeze(-1),dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        return torch.matmul(torch.transpose(attention,1,2), h).squeeze(1),attention

class SelfAttentionLayer2(nn.Module):
    def __init__(self, dim, da):
        super(SelfAttentionLayer2, self).__init__()
        self.dim = dim
        self.Wq = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.Wk = nn.Parameter(torch.zeros(self.dim, self.dim))
        nn.init.xavier_uniform_(self.Wq.data, gain=1.414)
        nn.init.xavier_uniform_(self.Wk.data, gain=1.414)
        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        N = h.shape[0]
        assert self.dim == h.shape[1]
        q = torch.matmul(h, self.Wq)
        k = torch.matmul(h, self.Wk)
        e = torch.matmul(q, k.t()) / math.sqrt(self.dim)
        attention = F.softmax(e, dim=1)
        attention = attention.mean(dim=0)
        x = torch.matmul(attention, h)
        return x

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

        def forward(self, input, memory, mask=None):
            bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

            input = self.dropout(input)
            memory = self.dropout(memory)

            input_dot = self.input_linear(input)
            memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
            cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
            att = input_dot + memory_dot + cross_dot
            if mask is not None:
                att = att - 1e30 * (1 - mask[:,None])

                weight_one = F.softmax(att, dim=-1)
                output_one = torch.bmm(weight_one, memory)
                weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
                output_two = torch.bmm(weight_two, input)
            return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        # self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        # self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        N = input.size()[0]
        # edge = adj.nonzero().t()
        edge = adj._indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        # edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_h = h[edge[1, :], :].t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1)).cuda())
        # e_rowsum: N x 1

        # edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        # self.attentions = [SpGraphAttentionLayer(nfeat,
        #                                          nhid,
        #                                          dropout=dropout,
        #                                          alpha=alpha,
        #                                          concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

        # self.out_att = SpGraphAttentionLayer(nhid * nheads,
        #                                      nclass,
        #                                      dropout=dropout,
        #                                      alpha=alpha,
        #                                      concat=False)
        self.out_att = SpGraphAttentionLayer(nhid,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        x = self.out_att(x, adj)
        return x
        # return F.log_softmax(x, dim=1)

def _add_neighbors(kg, g, seed_set, hop):
    tails_of_last_hop = seed_set
    for h in range(hop):
        next_tails_of_last_hop = []
        for entity in tails_of_last_hop:
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                g.add_edge(entity, tail_and_relation[1])
                if entity != tail_and_relation[1]:
                    next_tails_of_last_hop.append(tail_and_relation[1])
        tails_of_last_hop = next_tails_of_last_hop

# http://dbpedia.org/ontology/director


