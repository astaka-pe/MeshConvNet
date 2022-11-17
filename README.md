[[English]](README.md) [[日本語]](README_ja.md)

This page is under construction...
# Mesh Convolutional Network

The implementation of **Geometry-aware Mesh Convolutional Network** based on [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) [ICLR 2017].

## Environment
```
torch 1.7.0
torch-geometric 1.7.1 
scipy 1.6.2
numpy 1.19.2
```

## Usage
```
from util.layer import MeshConv

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
```

# Review of GCNConv [ICLR2017] 
## Graph Convolutional Layer

$$
\begin{align}
f(A,X;W) =& \sigma((I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) X W)\\
% =& \sigma((2I_N - \hat{L}) X W)\\
\rightarrow& \sigma(\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} X W)
\end{align}
$$

### Variable
- $A \in \{0, 1\}^{n\times n} $: Adjacency matrix
- $D \in \mathbb{R}^{n \times n}$: Degree matrix
- $L \in \mathbb{R}^{n \times n}$: Graph Laplacian matrix 
- $L^{sym} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}} = I_N - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$: Symmetrically normalized graph Laplacian
- $\hat{L} = \frac{2}{\lambda_{max}} L^{sym} - I_N$: Scaled graph Laplcian
- $X \in \mathbb{R}^{n \times d}$: Vertex feature matrix (Input to each layer)
- $W \in \mathbb{R}^{d \times d^{\prime}}$: Learnable weight matrix
- $\sigma$: Activation function (ex. ReLU, softmax)

### Intuitive understanding

#### Renormalization
$\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} X$

$\hat{A} = I + A$

![\hat{D}_{ii} = \sum_{j}{\hat{A}_{ij}}](https://latex.codecogs.com/gif.latex?\hat{D}_{ii}=\sum_{j}{\hat{A}_{ij}})

# Extention to Mesh Convolution

Replace **graph laplacian** $G$, which only encodes connectivity of mesh, to **mesh laplacian** $M$, which also encodes geometry of mesh.

## Variable
<!-- - $M \in \mathbb{R}^{n \times n}$: Mesh Laplacian matrix
- $\hat{M} = D^{-\frac{1}{2}} M D^{-\frac{1}{2}}$: Symmetrically normalized mesh Laplacian -->
- $L \in \mathbb{R}^{n \times n}$: Mesh Laplacian matrix
- $D_{ii} = L_{ii}$: Degree matrix (continuous)
- $A = D - L$: Adjacency matrix (continuous)

## [Mesh Laplacian](http://rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes)

<img src="docs/meshlaplacian.png" width="700">

## Renormalization

$\hat{A} = I + A$

![\hat{D}_{ii} = \sum_{j}{\hat{A}_{ij}}](https://latex.codecogs.com/gif.latex?\hat{D}_{ii}=\sum_{j}{\hat{A}_{ij}})

## Mesh Convolution
$$
\begin{align}
f(A,X;W) =& \sigma((I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}) X W)\\
% =& \sigma((2I_N - \hat{L}) X W)\\
\rightarrow& \sigma(\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} X W)
\end{align}
$$

# Experiment: Mesh Restoration

## Settings

Assign a 6-dimensional feature to each vertex and train MeshConv to restore an original mesh from a smoothed mesh.
Compare the performance between GCNConv and MeshConv.

- Learning rate: 0.001
- Epoch: 1000
- Metrix $\epsilon$ : MSELoss of vertex position

## Results

- MeshConv outperforms GCNConv.
- MeshConv can restore bumpy-sphere (bottom) more accurately than GCNConv.

<table>
  <tr>
    <td width="24%">Input</td>
    <td width="24%">GCNConv</td>
    <td width="24%">MeshConv</td>
    <td width="24%">Ground truth</td>
  </tr>
  <tr>
    <td width="24%"><img src="docs/input.png" width="100%"/></td>
    <td width="24%"><img src="docs/gcnconv.png" width="100%"/></td>
    <td width="24%"><img src="docs/meshconv.png" width="100%"/></td>
    <td width="24%"><img src="docs/groundtruth.png" width="100%"/></td>
  </tr>

  <tr>
    <td width="24%">---</td>
    <td width="24%">0.008221</td>
    <td width="24%">0.007452</td>
    <td width="24%">---</td>
  </tr>
  <tr>
    <td width="24%"><img src="docs/input2.png" width="100%"/></td>
    <td width="24%"><img src="docs/gcnconv2.png" width="100%"/></td>
    <td width="24%"><img src="docs/meshconv2.png" width="100%"/></td>
    <td width="24%"><img src="docs/groundtruth2.png" width="100%"/></td>
  </tr>

  <tr>
    <td width="24%">---</td>
    <td width="24%">0.09511</td>
    <td width="24%">0.05705</td>
    <td width="24%">---</td>
  </tr>
</table>


## Loss transition

### Dodecahedron
<img src="docs/loss.png" width=800>

### Bumpy-sphere
<img src="docs/loss2.png" width=800>
