import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj, dense_to_sparse


# the environment generator
class Graph_Editer(nn.Module):
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        # float16: required to fit (K,n,n) editor + Adagrad on 24GB GPUs (paper: RTX 4090 24G).
        self.B = nn.Parameter(torch.empty(K, n, n, dtype=torch.float16, device=device))
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, n, num_sample, k, edge_index, noise_level=0.8):
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(dtype=torch.float16, device=self.device)

        # Official code flips A.numel()*noise_level cells (~0.8*n^2), which densifies the graph
        # and OOMs GCN on Yelp. Keep noise_level=0.8 semantics relative to existing edges.
        num_edges_to_modify = max(1, int(edge_index.shape[1] * noise_level))
        indices_to_modify = torch.randint(0, A.numel(), (num_edges_to_modify,), device=self.device)
        values_to_modify = torch.randint(0, 2, (num_edges_to_modify,), dtype=torch.float16, device=self.device)
        A.view(-1)[indices_to_modify] = values_to_modify

        P = torch.softmax(Bk.float(), dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float16, device=self.device)
        col_idx = torch.arange(0, n, device=self.device).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        # Equivalent to A + M * ((1-A) - A) without materializing A_c.
        C = torch.where(M > 0, 1.0 - A, A)
        del A, M, P
        torch.cuda.empty_cache()
        edge_index = dense_to_sparse(C)[0]
        del C
        torch.cuda.empty_cache()

        sum_bk = torch.sum(Bk.float()[S, col_idx], dim=1)
        logsumexp_bk = torch.logsumexp(Bk.float(), dim=0)
        log_p = sum_bk - logsumexp_bk

        return edge_index, log_p
