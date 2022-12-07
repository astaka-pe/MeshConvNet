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

class ChebMeshConv(nn.Module):
    def __init__(self, in_channels, out_channels, mesh, k=1, bias=True):
        super(ChebMeshConv, self).__init__()
        self.coef_list = mesh.get_chebconv_coef(k=k)
        for i in range(k):
            self.register_buffer("F{}".format(i), self.coef_list[i])
        self.w = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x):
        for i in range(len(self.coef_list)):
            F = getattr(self, "F{}".format(i))
            if i == 0:
                out = torch.sparse.mm(F, x)
            else:
                out += torch.sparse.mm(F, x)
        out = self.w(out)
        return out