import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor


class TRACE(nn.Module):
    """Transcription-factor-aware Regulatory Architecture for Candidate-Edge prioritization."""

    def __init__(
        self,
        in_dim,
        h1_dim,
        h2_dim,
        out_dim,
        num_head1,
        num_head2,
        neg_slope,
        dropout,
        device,
        num_nodes,
        attention_mode="topology_only",
        decoder="bilinear",
        encoder_type="gat",
        cross_attn_lse_tau=0.7,
    ):
        super().__init__()
        if attention_mode not in {"dual", "regulatory_only", "topology_only", "feature_only"}:
            raise ValueError(f"Unsupported attention_mode: {attention_mode}")
        if decoder not in {"hybrid", "bilinear", "bilinear_bias", "calibrated_mlp", "role_bilinear", "local_cross_attn", "local_cross_attn_lse"}:
            raise ValueError(f"Unsupported decoder: {decoder}")
        if encoder_type not in {"gat", "sparse_transformer"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.neg_slope = neg_slope
        self.dropout = dropout
        self.device = device
        self.num_nodes = num_nodes
        self.attention_mode = attention_mode
        self.decoder = decoder
        self.encoder_type = encoder_type
        self.cross_attn_lse_tau = float(cross_attn_lse_tau)

        self.h1_dim = num_head1 * h1_dim
        self.h2_dim = num_head2 * h2_dim
        self.emb = None

        if encoder_type == "gat":
            self.Conv1 = nn.ModuleList(
                [
                    EdgeIdxAttentionSparse3(in_dim, h1_dim, num_nodes, neg_slope, attention_mode=attention_mode)
                    for _ in range(num_head1)
                ]
            )
            self.Conv2 = nn.ModuleList(
                [
                    EdgeIdxAttentionSparse3(self.h1_dim, h2_dim, num_nodes, neg_slope, attention_mode=attention_mode)
                    for _ in range(num_head2)
                ]
            )
        else:
            if attention_mode != "topology_only":
                raise ValueError("sparse_transformer encoder currently supports attention_mode=topology_only only.")
            self.Conv1 = nn.ModuleList(
                [
                    SparseTopologyTransformerLayer(
                        in_dim=in_dim,
                        head_dim=h1_dim,
                        num_heads=num_head1,
                        num_nodes=num_nodes,
                        dropout=dropout,
                        merge="concat",
                    )
                ]
            )
            self.Conv2 = nn.ModuleList(
                [
                    SparseTopologyTransformerLayer(
                        in_dim=self.h1_dim,
                        head_dim=h2_dim,
                        num_heads=num_head2,
                        num_nodes=num_nodes,
                        dropout=dropout,
                        merge="mean",
                    )
                ]
            )

        self.edge_bilin = nn.Bilinear(out_dim, out_dim, 1, bias=True)
        self.source_bias = nn.Embedding(num_nodes, 1)
        self.target_bias = nn.Embedding(num_nodes, 1)
        self.tf_role_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.target_role_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.context_q_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.context_k_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.context_v_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.context_bilin = nn.Bilinear(out_dim, out_dim, 1, bias=False)
        self.cross_attn_topology_beta = nn.Parameter(torch.tensor(0.0))
        self.cross_attn_mlp = nn.Sequential(
            nn.Linear(5 * out_dim + 2, 2 * out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * out_dim, 1),
        )
        self._target_context_cache = None
        self.emb_linear = nn.Linear(h2_dim, out_dim, bias=True)
        self.edge_hybrid_mlp = nn.Sequential(
            nn.Linear(4 * out_dim + 2, 2 * out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * out_dim, 1),
        )
        self.edge_calibration_mlp = nn.Sequential(
            nn.Linear(2 * out_dim + 1, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.Conv1:
            attention.reset_parameters()
        for attention in self.Conv2:
            attention.reset_parameters()
        nn.init.xavier_uniform_(self.emb_linear.weight, gain=1.414)
        if self.emb_linear.bias is not None:
            nn.init.zeros_(self.emb_linear.bias)
        nn.init.xavier_uniform_(self.edge_bilin.weight, gain=1.414)
        nn.init.zeros_(self.edge_bilin.bias)
        nn.init.zeros_(self.source_bias.weight)
        nn.init.zeros_(self.target_bias.weight)
        for module in list(self.tf_role_proj.modules()) + list(self.target_role_proj.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in [self.context_q_proj, self.context_k_proj, self.context_v_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.414)
        nn.init.xavier_uniform_(self.context_bilin.weight, gain=1.414)
        self.cross_attn_topology_beta.data.zero_()
        self._target_context_cache = None
        for module in self.cross_attn_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.edge_hybrid_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.edge_calibration_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x, adj):
        if self.encoder_type == "gat":
            x1 = torch.cat([att(x, adj) for att in self.Conv1], dim=1)
            x1 = F.elu(x1)
            out = torch.mean(torch.stack([att(x1, adj) for att in self.Conv2]), dim=0)
            return F.elu(out)

        x1 = F.elu(self.Conv1[0](x, adj))
        out = self.Conv2[0](x1, adj)
        return F.elu(out)

    def _get_target_context_cache(self, emb):
        device = emb.device
        cache = self._target_context_cache
        if cache is not None and cache[0].device == device:
            return cache

        layer = self.Conv1[0]
        row = getattr(layer, "topo_row", None)
        col = getattr(layer, "topo_col", None)
        weight = getattr(layer, "topo_w", None)
        if row is None or col is None or row.numel() == 0:
            context_idx = torch.arange(self.num_nodes, dtype=torch.long, device=device).view(self.num_nodes, 1)
            context_mask = torch.ones((self.num_nodes, 1), dtype=torch.bool, device=device)
            context_weight = torch.zeros((self.num_nodes, 1), dtype=emb.dtype, device=device)
            self._target_context_cache = (context_idx, context_mask, context_weight)
            return self._target_context_cache

        row_cpu = row.detach().cpu().tolist()
        col_cpu = col.detach().cpu().tolist()
        weight_cpu = weight.detach().cpu().tolist() if weight is not None else [0.0] * len(row_cpu)
        counts = [0] * self.num_nodes
        for target in row_cpu:
            counts[int(target)] += 1
        width = max(counts, default=0) + 1

        context_idx = torch.zeros((self.num_nodes, width), dtype=torch.long, device=device)
        context_mask = torch.zeros((self.num_nodes, width), dtype=torch.bool, device=device)
        context_weight = torch.zeros((self.num_nodes, width), dtype=emb.dtype, device=device)
        node_idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        context_idx[:, 0] = node_idx
        context_mask[:, 0] = True

        offsets = [1] * self.num_nodes
        for target, neighbor, topo_weight in zip(row_cpu, col_cpu, weight_cpu):
            target = int(target)
            pos = offsets[target]
            if pos >= width:
                continue
            context_idx[target, pos] = int(neighbor)
            context_mask[target, pos] = True
            context_weight[target, pos] = float(topo_weight)
            offsets[target] += 1

        self._target_context_cache = (context_idx, context_mask, context_weight)
        return self._target_context_cache

    def local_topology_cross_attention(self, emb, tf_emb, tg_idx):
        context_idx, context_mask, context_weight = self._get_target_context_cache(emb)
        edge_context_idx = context_idx[tg_idx]
        edge_context_mask = context_mask[tg_idx]
        edge_context_weight = context_weight[tg_idx]
        context_emb = emb[edge_context_idx]

        query = self.context_q_proj(tf_emb).unsqueeze(1)
        key = self.context_k_proj(context_emb)
        value = self.context_v_proj(context_emb)
        score = (query * key).sum(dim=-1) / math.sqrt(tf_emb.shape[-1])
        score = score + self.cross_attn_topology_beta * edge_context_weight
        score = score.masked_fill(~edge_context_mask, -torch.finfo(score.dtype).max)
        alpha = torch.softmax(score, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return (alpha.unsqueeze(-1) * value).sum(dim=1)

    def local_topology_lse_pooling(self, emb, tf_emb, tg_idx):
        context_idx, context_mask, context_weight = self._get_target_context_cache(emb)
        edge_context_idx = context_idx[tg_idx]
        edge_context_mask = context_mask[tg_idx]
        edge_context_weight = context_weight[tg_idx]
        context_emb = emb[edge_context_idx]

        query = self.context_q_proj(tf_emb).unsqueeze(1)
        key = self.context_k_proj(context_emb)
        value = self.context_v_proj(context_emb)
        attn_score = (query * key).sum(dim=-1) / math.sqrt(tf_emb.shape[-1])
        attn_score = attn_score + self.cross_attn_topology_beta * edge_context_weight
        tf_context = tf_emb.unsqueeze(1).expand_as(value)
        neighbor_score = self.context_bilin(tf_context, value).squeeze(-1)
        evidence = attn_score + neighbor_score
        evidence = evidence.masked_fill(~edge_context_mask, -torch.finfo(evidence.dtype).max)
        tau = max(self.cross_attn_lse_tau, 1e-6)
        lse_score = tau * torch.logsumexp(evidence / tau, dim=1)
        alpha = torch.softmax(evidence / tau, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        context = (alpha.unsqueeze(-1) * value).sum(dim=1)
        return context, lse_score.unsqueeze(-1)

    def decode(self, tf_emb, tg_emb, tf_idx=None, tg_idx=None, all_emb=None):
        bilinear_score = self.edge_bilin(tf_emb, tg_emb)
        if self.decoder == "bilinear":
            return bilinear_score.squeeze(-1)
        if self.decoder == "role_bilinear":
            tf_role = self.tf_role_proj(tf_emb)
            target_role = self.target_role_proj(tg_emb)
            return self.edge_bilin(tf_role, target_role).squeeze(-1)
        if self.decoder == "bilinear_bias":
            if tf_idx is None or tg_idx is None:
                raise ValueError("bilinear_bias decoder requires source and target node indices.")
            bias = self.source_bias(tf_idx) + self.target_bias(tg_idx)
            return (bilinear_score + bias).squeeze(-1)
        if self.decoder == "local_cross_attn":
            if all_emb is None or tg_idx is None:
                raise ValueError("local_cross_attn decoder requires full embeddings and target node indices.")
            context_emb = self.local_topology_cross_attention(all_emb, tf_emb, tg_idx)
            context_score = self.context_bilin(tf_emb, context_emb)
            pair_emb = torch.cat(
                [
                    tf_emb,
                    tg_emb,
                    context_emb,
                    torch.abs(tg_emb - context_emb),
                    tf_emb * context_emb,
                    bilinear_score,
                    context_score,
                ],
                dim=-1,
            )
            return self.cross_attn_mlp(pair_emb).squeeze(-1)
        if self.decoder == "local_cross_attn_lse":
            if all_emb is None or tg_idx is None:
                raise ValueError("local_cross_attn_lse decoder requires full embeddings and target node indices.")
            context_emb, context_score = self.local_topology_lse_pooling(all_emb, tf_emb, tg_idx)
            pair_emb = torch.cat(
                [
                    tf_emb,
                    tg_emb,
                    context_emb,
                    torch.abs(tg_emb - context_emb),
                    tf_emb * context_emb,
                    bilinear_score,
                    context_score,
                ],
                dim=-1,
            )
            return self.cross_attn_mlp(pair_emb).squeeze(-1)
        if self.decoder == "calibrated_mlp":
            pair_emb = torch.cat(
                [torch.abs(tf_emb - tg_emb), tf_emb * tg_emb, bilinear_score],
                dim=-1,
            )
            return self.edge_calibration_mlp(pair_emb).squeeze(-1)

        dot_score = (tf_emb * tg_emb).sum(dim=-1, keepdim=True)
        pair_emb = torch.cat(
            [tf_emb, tg_emb, torch.abs(tf_emb - tg_emb), tf_emb * tg_emb, bilinear_score, dot_score],
            dim=-1,
        )
        return self.edge_hybrid_mlp(pair_emb).squeeze(-1)

    def forward(self, x, adj, edge_idx):
        embed = self.encode(x, adj)
        emb = F.dropout(F.elu(self.emb_linear(embed)), p=self.dropout, training=self.training)
        self.emb = emb
        return self.decode(emb[edge_idx[0]], emb[edge_idx[1]], edge_idx[0], edge_idx[1], all_emb=emb)



class SparseTopologyTransformerLayer(nn.Module):
    def __init__(self, in_dim, head_dim, num_heads, num_nodes, dropout=0.1, merge="concat"):
        super().__init__()
        if merge not in {"concat", "mean"}:
            raise ValueError(f"Unsupported merge mode: {merge}")
        self.in_dim = in_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.merge = merge
        self.out_dim = head_dim * num_heads if merge == "concat" else head_dim

        proj_dim = head_dim * num_heads
        self.q_proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, proj_dim, bias=False)
        self.out_proj = nn.Linear(proj_dim, self.out_dim, bias=True)
        self.residual_proj = nn.Identity() if in_dim == self.out_dim else nn.Linear(in_dim, self.out_dim, bias=False)
        self.norm1 = nn.LayerNorm(self.out_dim)
        self.norm2 = nn.LayerNorm(self.out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.out_dim, 2 * self.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.out_dim, self.out_dim),
        )
        self.jaccard_beta = nn.Parameter(torch.zeros(num_heads))

        self.register_buffer("topo_row", None)
        self.register_buffer("topo_col", None)
        self.register_buffer("topo_w", None)
        self.topo_ready = False
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight, gain=1.414)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight, gain=1.414)
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.jaccard_beta)

    def precompute_topo_jaccard_edges(self, adj, topk=20, thr=0.2, undirected=True, exclude_1hop=False, device=None):
        import heapq

        if device is None:
            device = adj.device()

        row, col, _ = adj.coo()
        row = row.detach().cpu()
        col = col.detach().cpu()
        num_nodes = adj.sparse_sizes()[0]

        neighbors = [set() for _ in range(num_nodes)]
        for r, c in zip(row.tolist(), col.tolist()):
            neighbors[r].add(c)
            if undirected:
                neighbors[c].add(r)

        degrees = [len(neighbors[i]) for i in range(num_nodes)]
        topo_edges_row = []
        topo_edges_col = []
        topo_edges_w = []

        for src in range(num_nodes):
            src_neighbors = neighbors[src]
            if not src_neighbors:
                continue

            candidates = set()
            for neighbor in src_neighbors:
                candidates.update(neighbors[neighbor])
            candidates.discard(src)
            if exclude_1hop:
                candidates.difference_update(src_neighbors)
            if not candidates:
                continue

            heap = []
            for dst in candidates:
                dst_neighbors = neighbors[dst]
                if not dst_neighbors:
                    continue
                inter = len(src_neighbors.intersection(dst_neighbors))
                if inter == 0:
                    continue
                union = degrees[src] + degrees[dst] - inter
                jac = inter / (union + 1e-8)
                if jac < thr:
                    continue
                if len(heap) < topk:
                    heapq.heappush(heap, (jac, dst))
                elif jac > heap[0][0]:
                    heapq.heapreplace(heap, (jac, dst))

            heap.sort(reverse=True)
            for jac, dst in heap:
                topo_edges_row.append(src)
                topo_edges_col.append(dst)
                topo_edges_w.append(jac)

        self.topo_row = torch.tensor(topo_edges_row, dtype=torch.long, device=device)
        self.topo_col = torch.tensor(topo_edges_col, dtype=torch.long, device=device)
        self.topo_w = torch.tensor(topo_edges_w, dtype=torch.float, device=device)
        self.topo_ready = True

    def forward(self, x, adj: SparseTensor):
        if self.topo_ready and self.topo_row is not None:
            row = self.topo_row
            col = self.topo_col
            jw = self.topo_w
        else:
            row, col, _ = adj.coo()
            jw = None

        if row.numel() == 0:
            projected = self.residual_proj(x)
            return self.norm2(projected + F.dropout(self.ffn(projected), p=self.dropout, training=self.training))

        q = self.q_proj(x).view(self.num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(self.num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(self.num_nodes, self.num_heads, self.head_dim)

        score = (q[row] * k[col]).sum(dim=-1) / math.sqrt(self.head_dim)
        if jw is not None:
            score = score + jw.unsqueeze(-1) * self.jaccard_beta.unsqueeze(0)

        alpha = torch.stack(
            [softmax(score[:, head], row, num_nodes=self.num_nodes) for head in range(self.num_heads)],
            dim=1,
        )
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = v[col] * alpha.unsqueeze(-1)
        out = x.new_zeros((self.num_nodes, self.num_heads, self.head_dim))
        out.index_add_(0, row, msg)
        out = out.reshape(self.num_nodes, self.num_heads * self.head_dim)
        if self.merge == "mean":
            out = out.view(self.num_nodes, self.num_heads, self.head_dim).mean(dim=1)
        else:
            out = self.out_proj(out)

        h = self.norm1(self.residual_proj(x) + F.dropout(out, p=self.dropout, training=self.training))
        h_ffn = self.ffn(h)
        return self.norm2(h + F.dropout(h_ffn, p=self.dropout, training=self.training))


class EdgeIdxAttentionSparse3(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes, neg_slope=0.2, bias=True, attention_mode="topology_only"):
        super().__init__()
        if attention_mode not in {"dual", "regulatory_only", "topology_only", "feature_only"}:
            raise ValueError(f"Unsupported attention_mode: {attention_mode}")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neg_slope = neg_slope
        self.num_nodes = num_nodes
        self.attention_mode = attention_mode

        self.w = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.w2 = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.att = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        self.att2 = nn.Parameter(torch.zeros(size=(2 * out_dim, 1)))
        self.W = nn.Parameter(torch.FloatTensor(out_dim, out_dim))
        self.Q = nn.Parameter(torch.FloatTensor(out_dim, 1))

        self.register_buffer("topo_row", None)
        self.register_buffer("topo_col", None)
        self.register_buffer("topo_w", None)
        self.topo_ready = False
        self.jaccard_beta = nn.Parameter(torch.tensor(0.0))

        self.gate_linear = nn.Sequential(
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
        )

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.w2.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.att.data, gain=1.414)
        nn.init.xavier_uniform_(self.att2.data, gain=1.414)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.Q.data, gain=1.414)
        for module in self.gate_linear.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.414)
                if module.bias is not None:
                    module.bias.data.zero_()

    def precompute_topo_jaccard_edges(self, adj, topk=20, thr=0.2, undirected=True, exclude_1hop=False, device=None):
        import heapq

        if device is None:
            device = adj.device()

        row, col, _ = adj.coo()
        row = row.detach().cpu()
        col = col.detach().cpu()
        num_nodes = adj.sparse_sizes()[0]

        neighbors = [set() for _ in range(num_nodes)]
        for r, c in zip(row.tolist(), col.tolist()):
            neighbors[r].add(c)
            if undirected:
                neighbors[c].add(r)

        degrees = [len(neighbors[i]) for i in range(num_nodes)]
        topo_edges_row = []
        topo_edges_col = []
        topo_edges_w = []

        for src in range(num_nodes):
            src_neighbors = neighbors[src]
            if not src_neighbors:
                continue

            candidates = set()
            for neighbor in src_neighbors:
                candidates.update(neighbors[neighbor])
            candidates.discard(src)
            if exclude_1hop:
                candidates.difference_update(src_neighbors)
            if not candidates:
                continue

            heap = []
            for dst in candidates:
                dst_neighbors = neighbors[dst]
                if not dst_neighbors:
                    continue
                inter = len(src_neighbors.intersection(dst_neighbors))
                if inter == 0:
                    continue
                union = degrees[src] + degrees[dst] - inter
                jac = inter / (union + 1e-8)
                if jac < thr:
                    continue
                if len(heap) < topk:
                    heapq.heappush(heap, (jac, dst))
                elif jac > heap[0][0]:
                    heapq.heapreplace(heap, (jac, dst))

            heap.sort(reverse=True)
            for jac, dst in heap:
                topo_edges_row.append(src)
                topo_edges_col.append(dst)
                topo_edges_w.append(jac)

        self.topo_row = torch.tensor(topo_edges_row, dtype=torch.long, device=device)
        self.topo_col = torch.tensor(topo_edges_col, dtype=torch.long, device=device)
        self.topo_w = torch.tensor(topo_edges_w, dtype=torch.float, device=device)
        self.topo_ready = True

    def forward(self, x, adj: SparseTensor):
        num_nodes = self.num_nodes
        out_dim = self.out_dim
        row, col, _ = adj.coo()

        if self.attention_mode == "feature_only":
            out = x @ self.w2
            if self.bias is not None:
                out = out + self.bias
            return out

        if self.attention_mode in {"dual", "regulatory_only"}:
            h = x @ self.w
            att_src = self.att[:out_dim]
            att_dst = self.att[out_dim:]
            e = torch.exp(-(((h[row] @ att_src).squeeze(-1) - (h[col] @ att_dst).squeeze(-1)) ** 2))
            alpha = softmax(e, row, num_nodes=num_nodes)
            alpha = F.dropout(alpha, training=self.training, p=0.1)
            regulatory_msg = h[col] * alpha.unsqueeze(-1)
            out1 = torch.zeros_like(h)
            out1.index_add_(0, row, regulatory_msg)
        else:
            out1 = x.new_zeros((num_nodes, out_dim))

        if self.attention_mode in {"dual", "topology_only"}:
            h2 = x @ self.w2
            if self.topo_ready and self.topo_row is not None:
                row2 = self.topo_row
                col2 = self.topo_col
                jw = self.topo_w
            else:
                row2, col2 = row, col
                jw = None

            out2 = torch.zeros_like(h2)
            if row2.numel() > 0:
                att2_src = self.att2[:out_dim]
                att2_dst = self.att2[out_dim:]
                e2 = F.leaky_relu(
                    (h2[row2] @ att2_src).squeeze(-1) + (h2[col2] @ att2_dst).squeeze(-1),
                    negative_slope=self.neg_slope,
                )
                if jw is not None:
                    e2 = e2 + self.jaccard_beta * jw
                alpha2 = softmax(e2, row2, num_nodes=num_nodes)
                alpha2 = F.dropout(alpha2, training=self.training, p=0.1)
                topo_msg = h2[col2] * alpha2.unsqueeze(-1)
                out2.index_add_(0, row2, topo_msg)
        else:
            out2 = x.new_zeros((num_nodes, out_dim))

        if self.attention_mode == "regulatory_only":
            out = out1
        elif self.attention_mode == "topology_only":
            out = out2
        else:
            gate_input = torch.cat([out1, out2], dim=-1)
            gate = torch.sigmoid(self.gate_linear(gate_input))
            out = gate * out1 + (1.0 - gate) * out2

        if self.bias is not None:
            out = out + self.bias
        return out
