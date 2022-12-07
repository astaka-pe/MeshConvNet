import numpy as np
import torch
from functools import reduce
import scipy as sp
from sklearn.preprocessing import normalize

class Mesh:
    def __init__(self, path, build_code=False, build_mat=False, manifold=True):
        self.path = path
        self.vs, self.faces = self.fill_from_file(path)
        self.compute_face_normals()
        self.compute_face_center()
        self.device = 'cpu'
        self.simp = False

        if manifold:
            self.build_gemm() #self.edges, self.ve
            self.compute_vert_normals()
            self.build_mesh_lap()

    def fill_from_file(self, path):
        vs, faces = [], []
        f = open(path)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind) for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        faces = np.asarray(faces, dtype=int)

        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        return vs, faces

    def build_gemm(self):
        self.ve = [[] for _ in self.vs]
        self.vei = [[] for _ in self.vs]
        edge_nb = []
        sides = []
        edge2key = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(self.faces):
            faces_edges = []
            for i in range(3):
                cur_edge = (face[i], face[(i + 1) % 3])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1])
                    self.ve[edge[0]].append(edges_count)
                    self.ve[edge[1]].append(edges_count)
                    self.vei[edge[0]].append(0)
                    self.vei[edge[1]].append(1)
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = edge2key[faces_edges[(idx + 1) % 3]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
                nb_count[edge_key] += 2
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
        self.edges = np.array(edges, dtype=np.int32)
        self.gemm_edges = np.array(edge_nb, dtype=np.int64)
        self.sides = np.array(sides, dtype=np.int64)
        self.edges_count = edges_count

    def compute_face_normals(self):
        face_normals = np.cross(self.vs[self.faces[:, 1]] - self.vs[self.faces[:, 0]], self.vs[self.faces[:, 2]] - self.vs[self.faces[:, 0]])
        norm = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-24
        face_areas = 0.5 * np.sqrt((face_normals**2).sum(axis=1))
        face_normals /= norm
        self.fn, self.fa = face_normals, face_areas

    def compute_vert_normals(self):
        vert_normals = np.zeros((3, len(self.vs)))
        face_normals = self.fn
        faces = self.faces

        nv = len(self.vs)
        nf = len(faces)
        mat_rows = faces.reshape(-1)
        mat_cols = np.array([[i] * 3 for i in range(nf)]).reshape(-1)
        mat_vals = np.ones(len(mat_rows))
        f2v_mat = sp.sparse.csr_matrix((mat_vals, (mat_rows, mat_cols)), shape=(nv, nf))
        vert_normals = sp.sparse.csr_matrix.dot(f2v_mat, face_normals)
        vert_normals = normalize(vert_normals, norm='l2', axis=1)
        self.vn = vert_normals
    
    def compute_face_center(self):
        faces = self.faces
        vs = self.vs
        self.fc = np.sum(vs[faces], 1) / 3.0
    
    def build_vf(self):
        vf = [set() for _ in range(len(self.vs))]
        for i, f in enumerate(self.faces):
            vf[f[0]].add(i)
            vf[f[1]].add(i)
            vf[f[2]].add(i)
        self.vf = vf
    
    def build_adj_mat(self):
        edges = self.edges
        v2v_inds = edges.T
        v2v_inds = torch.from_numpy(np.concatenate([v2v_inds, v2v_inds[[1, 0]]], axis=1)).long()
        v2v_vals = torch.ones(v2v_inds.shape[1]).float()
        self.Adj = torch.sparse.FloatTensor(v2v_inds, v2v_vals, size=torch.Size([len(self.vs), len(self.vs)]))
        self.v_dims = torch.sum(self.Adj.to_dense(), axis=1)
        D_inds = torch.stack([torch.arange(len(self.vs)), torch.arange(len(self.vs))], dim=0).long()
        D_vals = 1 / (torch.pow(self.v_dims, 0.5) + 1.0e-12)
        self.D_minus_half = torch.sparse.FloatTensor(D_inds, D_vals, size=torch.Size([len(self.vs), len(self.vs)]))

    def build_mesh_lap(self):
        self.build_adj_mat()

        vs = self.vs
        edges = self.edges
        faces = self.faces
        
        e_dict = {}
        for e in edges:
            e0, e1 = min(e), max(e)
            e_dict[(e0, e1)] = []
        
        for f in faces:
            s = vs[f[1]] - vs[f[0]]
            t = vs[f[2]] - vs[f[1]]
            u = vs[f[0]] - vs[f[2]]
            cos_0 = np.inner(s, -u) / (np.linalg.norm(s) * np.linalg.norm(u))
            cos_1 = np.inner(t, -s) / (np.linalg.norm(t) * np.linalg.norm(s)) 
            cos_2 = np.inner(u, -t) / (np.linalg.norm(u) * np.linalg.norm(t))
            cot_0 = cos_0 / (np.sqrt(1 - cos_0 ** 2) + 1e-12)
            cot_1 = cos_1 / (np.sqrt(1 - cos_1 ** 2) + 1e-12)
            cot_2 = cos_2 / (np.sqrt(1 - cos_2 ** 2) + 1e-12)
            key_0 = (min(f[1], f[2]), max(f[1], f[2]))
            key_1 = (min(f[2], f[0]), max(f[2], f[0]))
            key_2 = (min(f[0], f[1]), max(f[0], f[1]))
            e_dict[key_0].append(cot_0)
            e_dict[key_1].append(cot_1)
            e_dict[key_2].append(cot_2)
        
        for e in e_dict:
            e_dict[e] = -0.5 * (e_dict[e][0] + e_dict[e][1])

        C_ind = [[], []]
        C_val = []
        ident = [0] * len(vs)
        for e in e_dict:
            C_ind[0].append(e[0])
            C_ind[1].append(e[1])
            C_ind[0].append(e[1])
            C_ind[1].append(e[0])
            C_val.append(e_dict[e])
            C_val.append(e_dict[e])
            ident[e[0]] += -1.0 * e_dict[e]
            ident[e[1]] += -1.0 * e_dict[e]
        Am_ind = torch.LongTensor(C_ind)
        Am_val = -1.0 * torch.FloatTensor(C_val)
        self.Am = torch.sparse.FloatTensor(Am_ind, Am_val, torch.Size([len(vs), len(vs)]))

        for i in range(len(vs)):
            C_ind[0].append(i)
            C_ind[1].append(i)
        
        C_val = C_val + ident
        C_ind = torch.LongTensor(C_ind)
        C_val = torch.FloatTensor(C_val)
        # cotangent matrix
        self.Lm = torch.sparse.FloatTensor(C_ind, C_val, torch.Size([len(vs), len(vs)]))
        self.Dm = torch.diag(torch.tensor(ident)).float().to_sparse()
        self.Lm_sym = torch.sparse.mm(torch.pow(self.Dm, -0.5), torch.sparse.mm(self.Lm, torch.pow(self.Dm, -0.5).to_dense())).to_sparse()
        #self.L = torch.sparse.mm(self.D_minus_half, torch.sparse.mm(C, self.D_minus_half.to_dense()))
        self.Am_I = (torch.eye(len(vs)) + self.Am).to_sparse()
        Dm_I_diag = torch.sum(self.Am_I.to_dense(), dim=1)
        self.Dm_I = torch.diag(Dm_I_diag).to_sparse()
        self.meshconvF = torch.sparse.mm(torch.pow(self.Dm_I, -0.5), torch.sparse.mm(self.Am_I, torch.pow(self.Dm_I, -0.5).to_dense())).to_sparse()
    
    def get_chebconv_coef(self, k=2):
        coef_list = []
        eig_max = torch.lobpcg(self.Lm_sym, k=1)[0][0]
        Lm_hat = -1.0 * torch.eye(len(self.vs)) + 2.0 * self.Lm_sym / eig_max.item()
        self.Lm_hat = Lm_hat.to_sparse()
        for i in range(k):
            if i == 0:
                coef = torch.eye(len(self.vs)).to_sparse()
                coef_list.append(coef)
            elif i == 1:
                coef_list.append(self.Lm_hat)
            else:
                coef = 2.0 * torch.sparse.mm(self.Lm_hat, coef_list[-1].to_dense()) - coef_list[-2]
                coef_list.append(coef.to_sparse())
        return coef_list

    def save(self, filename):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()

        with open(filename, 'w') as fp:
            # Write positions
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write('v {0:.8f} {1:.8f} {2:.8f}\n'.format(x, y, z))

            # Write indices
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0] + 1
                i1 = indices[i + 1] + 1
                i2 = indices[i + 2] + 1
                fp.write('f {0} {1} {2}\n'.format(i0, i1, i2))
    
    def save_as_ply(self, filename, fn):
        assert len(self.vs) > 0
        vertices = np.array(self.vs, dtype=np.float32).flatten()
        indices = np.array(self.faces, dtype=np.uint32).flatten()
        fnormals = np.array(fn, dtype=np.float32).flatten()

        with open(filename, 'w') as fp:
            # Write Header
            fp.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(self.vs)))
            fp.write("property float x\nproperty float y\nproperty float z\n")
            fp.write("element face {}\n".format(len(self.faces)))
            fp.write("property list uchar int vertex_indices\n")
            fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
            fp.write("end_header\n")
            for i in range(0, vertices.size, 3):
                x = vertices[i + 0]
                y = vertices[i + 1]
                z = vertices[i + 2]
                fp.write("{0:.6f} {1:.6f} {2:.6f}\n".format(x, y, z))
            
            for i in range(0, len(indices), 3):
                i0 = indices[i + 0]
                i1 = indices[i + 1]
                i2 = indices[i + 2]
                c0 = fnormals[i + 0]
                c1 = fnormals[i + 1]
                c2 = fnormals[i + 2]
                c0 = np.clip(int(255 * c0), 0, 255)
                c1 = np.clip(int(255 * c1), 0, 255)
                c2 = np.clip(int(255 * c2), 0, 255)
                c3 = 255
                fp.write("3 {0} {1} {2} {3} {4} {5} {6}\n".format(i0, i1, i2, c0, c1, c2, c3))