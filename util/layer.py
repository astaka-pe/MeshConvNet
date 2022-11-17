import torch
import torch.nn as nn


class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh, device="cuda:0"):
        super(MeshConv, self).__init__()
        self.F = mesh.meshconvF.to(device)
        self.w = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        out = torch.sparse.mm(self.F, x)     # (Nv, Cin)
        out = self.w(out)               # (Nv, Cout)
        return out
    
    def _get_filter(self, mesh_lap):
        lambda_max, _ = torch.lobpcg(mesh_lap, k=1)
        F = 2.0 * mesh_lap / lambda_max[0] - torch.eye(mesh_lap.shape[0])
        F = F.to_sparse()
        return F