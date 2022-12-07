import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import wandb

from util.mesh import Mesh
from util.model import MeshNet, ChebMeshNet, GCN

def get_parser():
    parser = argparse.ArgumentParser(description="MeshConv")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("--smooth", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=1000)
    args = parser.parse_args()

    return args

def torch_fix_seed(seed=314):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def main():
    args = get_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mesh = Mesh(args.input)
    o_mesh = Mesh(args.input)
    s_mesh = Mesh(args.smooth)
    s_vs = torch.from_numpy(s_mesh.vs).to(device)
    edge_index = torch.tensor(mesh.edges.T, dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index[[1,0],:]], dim=1).to(device)
    x = np.concatenate([s_mesh.vs, s_mesh.vn], axis=1)
    #x = np.random.randn(len(mesh.vs), 8)
    x = torch.tensor(x, requires_grad=True).float().to(device)
    
    torch_fix_seed()
    # net = MeshNet(mesh).to(device)
    net = ChebMeshNet(mesh).to(device)
    #net = GCN(edge_index).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    mseloss = nn.MSELoss()
    os.makedirs("data/output", exist_ok=True)

    wandb.init(project="MeshConv")
    for i in range(1, args.epoch+1):
        net.train()
        optimizer.zero_grad()
        dis = net(x)
        out = s_vs + dis
        loss = mseloss(out.double(), torch.from_numpy(mesh.vs).double().to(device))
        wandb.log({"loss": loss}, step=i)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print("epoch: {:03} / loss: {:4f}".format(i, loss))
            o_mesh.vs = out.detach().to("cpu").numpy().copy()
            o_mesh.save("data/output/{}.obj".format(i))
        
if __name__ == "__main__":
    main()