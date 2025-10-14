import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def link_logits(self, z, edge_label_index):

        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, att_heads, dropout=0.5):
        super().__init__()
        self.att1 = GATConv(in_dim, hidden_dim, att_heads, "mean")
        self.att2 = GATConv(hidden_dim, out_dim, att_heads, "mean")
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.att1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.att2(x, edge_index)
        return x

    def link_logits(z, edge_label_index):

        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)