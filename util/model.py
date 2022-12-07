import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Sequential

from util.layer import MeshConv, ChebMeshConv


class MeshNet(nn.Module):
    def __init__(self, mesh):
        super(MeshNet, self).__init__()
        self.model = nn.Sequential(
            MeshConv(6, 32, mesh),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            MeshConv(32, 128, mesh),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            MeshConv(128, 128, mesh),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            MeshConv(128, 32, mesh),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            MeshConv(32, 16, mesh),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        out = self.model(x)
        return out

class ChebMeshNet(nn.Module):
    def __init__(self, mesh):
        super(ChebMeshNet, self).__init__()
        self.model = nn.Sequential(
            ChebMeshConv(6, 32, mesh, k=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            ChebMeshConv(32, 128, mesh, k=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            ChebMeshConv(128, 128, mesh, k=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            ChebMeshConv(128, 32, mesh, k=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            ChebMeshConv(32, 16, mesh, k=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        out = self.model(x)
        return out

class GCN(nn.Module):
    def __init__(self, edge_index):
        super(GCN, self).__init__()
        self.edge_index = edge_index
        self.model = Sequential("x, edge_index", [
            (GCNConv(6, 32), "x, edge_index -> x"),
            (nn.BatchNorm1d(32), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (GCNConv(32, 128), "x, edge_index -> x"),
            (nn.BatchNorm1d(128), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (GCNConv(128, 128), "x, edge_index -> x"),
            (nn.BatchNorm1d(128), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (GCNConv(128, 32), "x, edge_index -> x"),
            (nn.BatchNorm1d(32), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (GCNConv(32, 16), "x, edge_index -> x"),
            (nn.BatchNorm1d(16), "x -> x"),
            (nn.LeakyReLU(), "x -> x"),
            (nn.Linear(16, 3), "x -> x"),
        ])
    
    def forward(self, x):
        out = self.model(x, self.edge_index)
        return out