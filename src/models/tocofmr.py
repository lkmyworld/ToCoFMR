import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class ToCoFMR(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.embedding_dim = int(config["embedding_dim"])
        self.n_mm_layers = int(config["n_mm_layers"])
        self.n_ui_layers = int(config["n_ui_layers"])
        self.mm_knn_k = int(config["mm_knn_k"])
        self.co_knn_k = int(config["co_knn_k"])
        self.mm_image_weight = float(config["mm_image_weight"])
        self.cl_weight = float(config["cl_weight"])
        self.temperature = float(config["temperature"])
        self.missing_modal = bool(config["missing_modal"])
        self.flow_weight = float(config["flow_weight"])
        self.topology_weight = float(config["topology_weight"])
        self.graph_fusion_weight = float(config["graph_fusion_weight"])
        self.graph_warmup_epochs = int(config["graph_warmup_epochs"])
        self.flow_steps = int(config["flow_steps"])
        self.flow_hidden_dim = int(config["flow_hidden_dim"])

        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.interaction_matrix_csr = dataset.inter_matrix(form="csr").astype(np.float32)
        self.n_nodes = self.n_users + self.n_items
        if self.missing_modal:
            self.preprocess_missing_modal(config)

        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)

        self.R = torch.sparse_coo_tensor(self.edge_indices, self.edge_values, torch.Size((self.n_users, self.n_items))).coalesce().to(self.device)
        self.co_item_adj = self.build_collaborative_item_adj().to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        obs_v = torch.ones(self.n_items, dtype=torch.float32)
        obs_t = torch.ones(self.n_items, dtype=torch.float32)
        if self.missing_modal:
            obs_v[self.missing_items_v] = 0.0
            obs_t[self.missing_items_t] = 0.0
        self.obs_v_mask = obs_v.to(self.device)
        self.obs_t_mask = obs_t.to(self.device)

        self.observed_image_adj = self.build_masked_knn_adj(self.v_feat, self.obs_v_mask).to(self.device)
        self.observed_text_adj = self.build_masked_knn_adj(self.t_feat, self.obs_t_mask).to(self.device)

        self.image_proj = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.text_proj = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.image_proj.apply(_init)
        self.text_proj.apply(_init)

        self.visual_source_map = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.text_source_map = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        nn.init.eye_(self.visual_source_map.weight)
        nn.init.eye_(self.text_source_map.weight)

        self.image_flow = ConditionalRectifiedFlow(self.embedding_dim, self.embedding_dim, self.flow_hidden_dim, self.flow_steps)
        self.text_flow = ConditionalRectifiedFlow(self.embedding_dim, self.embedding_dim, self.flow_hidden_dim, self.flow_steps)
        self.image_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.text_gate = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.image_gate.apply(_init)
        self.text_gate.apply(_init)

        noise_v_generator = torch.Generator(device="cpu").manual_seed(20260311)
        noise_t_generator = torch.Generator(device="cpu").manual_seed(20260312)
        self.source_noise_v = torch.randn(self.n_items, self.embedding_dim, generator=noise_v_generator).to(self.device)
        self.source_noise_t = torch.randn(self.n_items, self.embedding_dim, generator=noise_t_generator).to(self.device)

        self.cached_image_item = torch.zeros(self.n_items, self.embedding_dim, device=self.device)
        self.cached_text_item = torch.zeros(self.n_items, self.embedding_dim, device=self.device)

        self.cached_image_adj = self.observed_image_adj
        self.cached_text_adj = self.observed_text_adj

        self.graph_epoch = 0

    def preprocess_missing_modal(self, config):
        self.missing_modal = config['missing_modal']
        self.missing_ratio = config['missing_ratio']
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])

        missing_filename = f"missing_items_{config['missing_ratio']}.npy"

        missing_path = os.path.join(dataset_path, missing_filename)
        if not os.path.isfile(missing_path):
            raise FileNotFoundError(f"Missing-modality file not found: {missing_path}")

        missing = np.load(missing_path, allow_pickle=True).item()
        self.missing_items_t = np.concatenate((missing["all"], missing["t"])).astype(np.int64)
        self.missing_items_v = np.concatenate((missing["all"], missing["v"])).astype(np.int64)
        self.v_feat[self.missing_items_v] = 0
        self.t_feat[self.missing_items_t] = 0

    def pre_epoch_processing(self):
        self.refresh_completion_graph()

    def compute_weighted_knn_graph(self, similarity):
        topk = min(self.mm_knn_k, similarity.shape[0])

        _, neighbours = torch.topk(similarity, topk, dim=-1)

        rows = torch.arange(similarity.shape[0], device=similarity.device).unsqueeze(1).expand(-1, topk).reshape(-1)
        cols = neighbours.reshape(-1)
        values = torch.clamp(similarity[rows, cols], min=0.0)
        row_sum = torch.zeros(similarity.shape[0], device=similarity.device)
        col_sum = torch.zeros(similarity.shape[0], device=similarity.device)
        row_sum.index_add_(0, rows, values)
        col_sum.index_add_(0, cols, values)
        values = values / torch.sqrt((row_sum[rows] + 1e-7) * (col_sum[cols] + 1e-7))
        return torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), values, torch.Size((similarity.shape[0], similarity.shape[0]))).coalesce()

    @torch.no_grad()
    def build_knn_adj(self, features):
        normalized = F.normalize(features, dim=-1, eps=1e-8)
        similarity = torch.clamp(normalized @ normalized.t(), min=0.0)
        similarity.fill_diagonal_(1.0)
        return self.compute_weighted_knn_graph(similarity)

    @torch.no_grad()
    def build_masked_knn_adj(self, features, observed_mask):
        normalized = F.normalize(features, dim=-1, eps=1e-8)
        similarity = torch.clamp(normalized @ normalized.t(), min=0.0)
        missing = observed_mask <= 0.0
        if missing.any():
            similarity[missing, :] = 0.0
            similarity[:, missing] = 0.0
            similarity[missing, missing] = 1.0
        similarity.fill_diagonal_(1.0)
        return self.compute_weighted_knn_graph(similarity)

    @staticmethod
    def fuse_graphs(observed_graph, completed_graph, mix_weight):
        observed_graph = observed_graph.coalesce()
        completed_graph = completed_graph.coalesce()
        indices = torch.cat([observed_graph.indices(), completed_graph.indices()], dim=1)
        values = torch.cat([(1.0 - mix_weight) * observed_graph.values(), mix_weight * completed_graph.values()])
        return torch.sparse_coo_tensor(indices, values, observed_graph.shape, device=values.device).coalesce()

    def build_collaborative_item_adj(self):
        co_matrix = self.interaction_matrix_csr.transpose().dot(self.interaction_matrix_csr).tolil()
        co_matrix.setdiag(0.0)
        co_matrix = co_matrix.tocsr()
        rows, cols, values = [], [], []
        for item_id in range(self.n_items):
            start, end = co_matrix.indptr[item_id:item_id + 2]
            item_cols, item_values = co_matrix.indices[start:end], co_matrix.data[start:end]
            if item_values.size == 0:
                rows.append(item_id)
                cols.append(item_id)
                values.append(1.0)
                continue
            topk = min(self.co_knn_k, item_values.size)
            top_indices = np.argpartition(-item_values, topk - 1)[:topk]
            rows.extend([item_id] * topk)
            cols.extend(item_cols[top_indices].tolist())
            values.extend(item_values[top_indices].astype(np.float32).tolist())
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        row_sum = np.zeros(self.n_items, dtype=np.float32)
        col_sum = np.zeros(self.n_items, dtype=np.float32)
        np.add.at(row_sum, rows, values)
        np.add.at(col_sum, cols, values)
        values = values / np.sqrt((row_sum[rows] + 1e-7) * (col_sum[cols] + 1e-7))
        return torch.sparse_coo_tensor(torch.LongTensor(np.stack([rows, cols], axis=0)), torch.FloatTensor(values), torch.Size((self.n_items, self.n_items))).coalesce()

    def get_norm_adj_mat(self):
        adjacency = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        transposed = self.interaction_matrix.transpose()
        adjacency.update(dict(zip(zip(self.interaction_matrix.row, self.interaction_matrix.col + self.n_users), [1] * self.interaction_matrix.nnz)))
        adjacency.update(dict(zip(zip(transposed.row + self.n_users, transposed.col), [1] * transposed.nnz)))
        degree = np.asarray((adjacency > 0).sum(axis=1)).reshape(-1) + 1e-7
        degree = np.power(degree, -0.5)
        normalized = sp.coo_matrix(sp.diags(degree) * adjacency * sp.diags(degree))
        return torch.sparse_coo_tensor(torch.LongTensor(np.array([normalized.row, normalized.col])), torch.FloatTensor(normalized.data), torch.Size((self.n_nodes, self.n_nodes)))

    def _normalize_adj_m(self, indices, shape):
        adjacency = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), shape)
        row_sum = 1e-7 + torch.sparse.sum(adjacency, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adjacency.t(), -1).to_dense()
        return torch.pow(row_sum, -0.5)[indices[0]] * torch.pow(col_sum, -0.5)[indices[1]]

    def get_edge_info(self):
        indices = torch.stack([torch.from_numpy(self.interaction_matrix.row), torch.from_numpy(self.interaction_matrix.col)]).long()
        return indices, self._normalize_adj_m(indices, torch.Size((self.n_users, self.n_items)))

    def bpr_loss(self, users, positive_items, negative_items):
        return -torch.mean(F.logsigmoid(torch.sum(users * positive_items, dim=1) - torch.sum(users * negative_items, dim=1)))

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        positive = torch.exp(torch.sum(view1 * view2, dim=-1) / temperature)
        total = torch.exp((view1 @ view2.t()) / temperature).sum(dim=1)
        return torch.mean(-torch.log(positive / total.clamp(min=1e-8)))

    def modalities_encoder(self):
        return F.normalize(self.image_proj(self.v_feat), dim=-1, eps=1e-8) * self.obs_v_mask.unsqueeze(-1), F.normalize(self.text_proj(self.t_feat), dim=-1, eps=1e-8) * self.obs_t_mask.unsqueeze(-1)

    def collaborative_item(self):
        co_item = self.item_id_embedding.weight
        for _ in range(self.n_mm_layers):
            co_item = torch.sparse.mm(self.co_item_adj, co_item)
        return co_item

    def propagate(self, item, graph):
        for _ in range(self.n_mm_layers):
            item = torch.sparse.mm(graph, item)
        return item

    @torch.no_grad()
    def refresh_completion_graph(self):
        raise NotImplementedError(
            "refresh_completion_graph implementation will be released in the future."
        )

    def complete_modalities(self, image, text):
        raise NotImplementedError(
            "complete_modalities implementation will be released in the future."
        )

    def source_distribution(self, image, text):
        raise NotImplementedError(
            "source_distribution implementation will be released in the future."
        )

    def forward(self, adjacency, train=False):
        user_id = self.user_embedding.weight
        item_id = self.item_id_embedding.weight
        collaborative = torch.cat([user_id, item_id], dim=0)
        layers = [collaborative]
        for _ in range(self.n_ui_layers):
            collaborative = torch.sparse.mm(adjacency, collaborative)
            layers.append(collaborative)
        cf_user, cf_item = torch.split(torch.stack(layers, dim=1).mean(dim=1), [self.n_users, self.n_items], dim=0, )

        current_image, current_text = self.modalities_encoder()
        obs_v = self.obs_v_mask.unsqueeze(-1)
        obs_t = self.obs_t_mask.unsqueeze(-1)
        image_for_rec = (obs_v * current_image + (1.0 - obs_v) * self.cached_image_item.detach())
        text_for_rec = (obs_t * current_text + (1.0 - obs_t) * self.cached_text_item.detach())

        seed_item = self.collaborative_item()
        image_item = seed_item * self.image_gate(image_for_rec)
        text_item = seed_item * self.text_gate(text_for_rec)
        image_item = self.propagate(image_item, self.cached_image_adj)
        text_item = self.propagate(text_item, self.cached_text_adj)

        mm_item = (self.mm_image_weight * image_item + (1.0 - self.mm_image_weight) * text_item)
        side_user = torch.sparse.mm(self.R, mm_item)
        user = cf_user + side_user
        item = cf_item + mm_item
        if train:
            return user, item, cf_user, cf_item, side_user, mm_item
        return user, item

    def flow_loss(self, items):
        raise NotImplementedError(
            "flow_loss implementation will be released in the future."
        )

    def calculate_loss(self, interaction):
        users, positive_items, negative_items = interaction[:3]
        all_user, all_item, cf_user, cf_item, side_user, side_item = self.forward(self.norm_adj, train=True)
        recommendation = self.bpr_loss(all_user[users], all_item[positive_items], all_item[negative_items])
        contrastive = (self.InfoNCE(side_item[positive_items], cf_item[positive_items], self.temperature) + self.InfoNCE(side_user[users], cf_user[users], self.temperature))

        items = torch.unique(torch.cat([positive_items, negative_items], dim=0))

        return (recommendation + self.cl_weight * contrastive + self.flow_loss(items))

    @torch.no_grad()
    def full_sort_predict(self, interaction):
        users = interaction[0]
        all_user, all_item = self.forward(self.norm_adj)
        return all_user[users] @ all_item.t()


def _init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConditionalRectifiedFlow(nn.Module):
    def __init__(self, dim, context_dim, hidden_dim, steps):
        super().__init__()
        raise NotImplementedError(
            "ConditionalRectifiedFlow implementation will be released in the future."
        )
