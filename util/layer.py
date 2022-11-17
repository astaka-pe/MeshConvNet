import torch
import torch.nn as nn


class MeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh, bias=True):
        super(MeshConv, self).__init__()
        self.register_buffer("F", mesh.meshconvF)
        self.w = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x):
        out = torch.sparse.mm(self.F, x)     # (Nv, Cin)
        out = self.w(out)                    # (Nv, Cout)
        return out