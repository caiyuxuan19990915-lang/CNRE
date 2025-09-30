import os.path
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from data_setp import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss


def build_faiss_index(item_embeddings: torch.Tensor):
    item_embeddings_np = item_embeddings.detach().cpu().contiguous().numpy()
    d = item_embeddings_np.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(item_embeddings_np)
    return index


class GraphEncoder(nn.Module):
    def __init__(self, layer_nums, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layer_nums)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        result = [x]
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
        return torch.sum(torch.stack(result, dim=0), dim=0)


class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()

    def forward(self, adj, embeds):
        return adj @ (adj.T @ embeds)


class Hyper_behavior_gcn(nn.Module):
    def __init__(self, args, n_users, n_items, layer_nums=2):
        super(Hyper_behavior_gcn, self).__init__()
        latdim = args.embedding_size
        self.n_users, self.n_items = n_users, n_items
        self.hgnnLayer = HGNNLayer()
        self.uHyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(latdim, args.hyper_nums)))
        self.iHyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(latdim, args.hyper_nums)))
        self.dropout1 = nn.Dropout(p=args.node_dropout)  # hyper_dropout in original
        self.dropout2 = nn.Dropout(p=args.node_dropout)  # hyper_dropout in original

    def forward(self, embeds):
        u_embeds, i_embeds = torch.split(embeds, [self.n_users + 1, self.n_items + 1])
        H_u = self.dropout1(u_embeds @ self.uHyper)
        H_i = self.dropout2(i_embeds @ self.iHyper)
        u_sem_embeds = self.hgnnLayer(H_u, u_embeds)
        i_sem_embeds = self.hgnnLayer(H_i, i_embeds)
        return torch.cat([u_sem_embeds, i_sem_embeds], dim=0)


class ConfirmatoryConjunction(nn.Module):
    def __init__(self, embedding_size):
        super(ConfirmatoryConjunction, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_size * 3, embedding_size), nn.LeakyReLU(),
                                 nn.Linear(embedding_size, embedding_size))

    def forward(self, user_emb, target_item_emb, evidence_item_emb):
        return self.mlp(torch.cat([user_emb, target_item_emb, evidence_item_emb], dim=-1))


class SupplementaryDisjunction(nn.Module):
    def __init__(self, embedding_size):
        super(SupplementaryDisjunction, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_size * 3, embedding_size), nn.LeakyReLU(),
                                 nn.Linear(embedding_size, embedding_size))

    def forward(self, user_emb, target_item_emb, evidence_item_emb):
        return self.mlp(torch.cat([user_emb, target_item_emb, evidence_item_emb], dim=-1))


class Prediction(nn.Module):
    def __init__(self, embedding_size):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_size // 2, 1)
        )

    def forward(self, item_embedding):
        return self.mlp(item_embedding)


class Model(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(Model, self).__init__()
        self.args = args
        self.dataset = dataset
        self.behaviors = args.behaviors
        self.n_behaviors = len(self.behaviors)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.embedding_size = args.embedding_size
        self.device = args.device
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.global_graph_encoder = GraphEncoder(args.layers, self.embedding_size, args.node_dropout)
        self.behavior_graph_encoder = nn.ModuleDict(
            {b: GraphEncoder(args.layers_nums[i], self.embedding_size, args.node_dropout) for i, b in
             enumerate(self.behaviors)})
        self.behavior_hyper_graph_encoder = nn.ModuleDict(
            {b: Hyper_behavior_gcn(args, self.n_users, self.n_items, 1) for b in self.behaviors})
        self.faiss_indexes_gcn = {}
        self.faiss_indexes_hyper = {}
        self.confirmatory_conjunction = ConfirmatoryConjunction(self.embedding_size)
        self.supplementary_disjunction = SupplementaryDisjunction(self.embedding_size)
        self.prediction = Prediction(self.embedding_size)
        self.confidence_threshold = args.conf_threshold
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.reg_weight = args.reg_weight
        self.storage_final_user_embeddings = None
        self.storage_final_item_embeddings = None
        self.storage_gcn_item_embeds_dict = None
        self.storage_hyper_item_embeds_dict = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None: nn.init.constant_(module.bias, 0)

    def hierarchical_preference_propagation(self, return_intermediate=False):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        global_embeddings = self.global_graph_encoder(all_embeddings, self.dataset.all_edge_index.to(self.device))

        gcn_item_embeds_dict, hyper_item_embeds_dict = {}, {}
        all_user_embeds, all_item_embeds = [], []
        cascading_embeddings = global_embeddings

        for i, behavior in enumerate(self.behaviors):
            temp_cascading_embeddings = cascading_embeddings
            indices = self.dataset.edge_index[behavior].to(self.device)

            behavior_gcn_embeddings = self.behavior_graph_encoder[behavior](cascading_embeddings, indices)
            behavior_hyper_embeddings = self.behavior_hyper_graph_encoder[behavior](behavior_gcn_embeddings)

            if return_intermediate:
                _, gcn_item_embeds = torch.split(behavior_gcn_embeddings, [self.n_users + 1, self.n_items + 1])
                _, hyper_item_embeds = torch.split(behavior_hyper_embeddings, [self.n_users + 1, self.n_items + 1])
                gcn_item_embeds_dict[behavior] = gcn_item_embeds
                hyper_item_embeds_dict[behavior] = hyper_item_embeds

            e_col, e_sem = behavior_gcn_embeddings, behavior_hyper_embeddings
            numerator = torch.sum(e_sem * e_col, dim=-1, keepdim=True)
            denominator = torch.sum(e_col.pow(2), dim=-1, keepdim=True) + 1e-8
            p_col = (numerator / denominator) * e_col
            cascading_embeddings = temp_cascading_embeddings + behavior_gcn_embeddings + p_col
            user_embedding, item_embedding = torch.split(cascading_embeddings, [self.n_users + 1, self.n_items + 1])
            all_user_embeds.append(user_embedding)
            all_item_embeds.append(item_embedding)

        final_all_user_embeds = torch.stack(all_user_embeds, dim=1)
        final_all_item_embeds = torch.stack(all_item_embeds, dim=1)

        self.storage_final_user_embeddings = final_all_user_embeds
        self.storage_final_item_embeddings = final_all_item_embeds
        if return_intermediate:
            self.storage_gcn_item_embeds_dict = gcn_item_embeds_dict
            self.storage_hyper_item_embeds_dict = hyper_item_embeds_dict

        return (final_all_user_embeds, final_all_item_embeds, gcn_item_embeds_dict,
                hyper_item_embeds_dict) if return_intermediate else (final_all_user_embeds, final_all_item_embeds)

    def update_faiss_indexes(self):
        self.eval()
        with torch.no_grad():
            self.hierarchical_preference_propagation(return_intermediate=True)
            gcn_item_embeds = self.storage_gcn_item_embeds_dict
            hyper_item_embeds = self.storage_hyper_item_embeds_dict

            for behavior in self.behaviors:
                self.faiss_indexes_gcn[behavior] = build_faiss_index(F.normalize(gcn_item_embeds[behavior], p=2, dim=1))
            for behavior in self.behaviors[:-1]:
                if behavior in hyper_item_embeds:
                    self.faiss_indexes_hyper[behavior] = build_faiss_index(
                        F.normalize(hyper_item_embeds[behavior], p=2, dim=1))
        self.train()

    def neuro_symbolic_reasoning(self, users, items, histories, target_behavior_idx, all_user_embeds,
                                 all_item_embeds, gcn_embeds, hyper_embeds):

        causal_mediators = all_item_embeds[items, target_behavior_idx]
        behavior_name = self.behaviors[target_behavior_idx]
        prev_b_name = self.behaviors[target_behavior_idx - 1] if target_behavior_idx > 0 else self.behaviors[0]

        scene1_mask = (histories[:, -1] == 1)
        scene3_mask = (~(histories[:, -1] == 1)) & (histories.sum(dim=1) <= 1)
        scene2_mask = ~(scene1_mask | scene3_mask)

        if scene2_mask.any():
            s2_users, s2_items = users[scene2_mask], items[scene2_mask]
            s2_base_user_emb = all_user_embeds[s2_users, target_behavior_idx]
            s2_base_item_emb = causal_mediators[scene2_mask]
            confidence = torch.sigmoid(torch.sum(s2_base_user_emb * s2_base_item_emb, dim=1))
            low_conf_mask = confidence < self.confidence_threshold

            if low_conf_mask.any():
                s2_low_users = s2_users[low_conf_mask]
                s2_low_user_emb = all_user_embeds[s2_low_users, target_behavior_idx]
                s2_low_items = s2_items[low_conf_mask]
                s2_low_item_gcn_emb = gcn_embeds[prev_b_name][s2_low_items]

                query_emb_np = F.normalize(s2_low_item_gcn_emb, p=2, dim=1).detach().cpu().numpy()
                _, evidence_indices = self.faiss_indexes_gcn[prev_b_name].search(query_emb_np, k=2)

                evidence_indices = torch.tensor(evidence_indices[:, 1], device=self.device)
                evidence_gcn_emb = gcn_embeds[prev_b_name][evidence_indices]
                causal_mediator = self.confirmatory_conjunction(s2_low_user_emb, s2_low_item_gcn_emb, evidence_gcn_emb)
                causal_mediators[scene2_mask.nonzero(as_tuple=True)[0][low_conf_mask]] = causal_mediator

        if scene3_mask.any() and prev_b_name in self.faiss_indexes_hyper:
            s3_users, s3_items = users[scene3_mask], items[scene3_mask]
            s3_user_emb = all_user_embeds[s3_users, target_behavior_idx]
            s3_item_emb = all_item_embeds[s3_items, target_behavior_idx]
            s3_hyper_item_emb = hyper_embeds[prev_b_name][s3_items]

            query_emb_np = F.normalize(s3_hyper_item_emb, p=2, dim=1).detach().cpu().numpy()
            _, evidence_indices = self.faiss_indexes_hyper[prev_b_name].search(query_emb_np, k=2)

            evidence_indices = torch.tensor(evidence_indices[:, 1], device=self.device)
            evidence_hyper_emb = hyper_embeds[prev_b_name][evidence_indices]
            causal_mediator = self.supplementary_disjunction(s3_user_emb, s3_item_emb, evidence_hyper_emb)
            causal_mediators[scene3_mask] = causal_mediator

        return causal_mediators

    def forward(self, batch_data):
        users, pos_items_list, neg_items_list, histories_list = batch_data

        final_all_user_embeds, final_all_item_embeds, gcn_item_embeds_dict, hyper_item_embeds_dict = self.hierarchical_preference_propagation(
            return_intermediate=True)

        total_loss = 0.0
        for i in range(self.n_behaviors):
            pos_items = pos_items_list[:, i]
            neg_items = neg_items_list[:, i]
            pos_histories = histories_list[:, i, :]

            valid_mask = pos_items != 0
            if not valid_mask.any():
                continue

            neg_histories = torch.zeros_like(pos_histories[valid_mask])

            mediator_pos = self.neuro_symbolic_reasoning(users[valid_mask], pos_items[valid_mask],
                                                         pos_histories[valid_mask], i, final_all_user_embeds,
                                                         final_all_item_embeds, gcn_item_embeds_dict,
                                                         hyper_item_embeds_dict)
            mediator_neg = self.neuro_symbolic_reasoning(users[valid_mask], neg_items[valid_mask], neg_histories, i,
                                                         final_all_user_embeds, final_all_item_embeds,
                                                         gcn_item_embeds_dict, hyper_item_embeds_dict)

            pos_scores = self.prediction(mediator_pos).squeeze()
            neg_scores = self.prediction(mediator_neg).squeeze()

            loss_b = self.bpr_loss(pos_scores, neg_scores)
            total_loss += loss_b

        reg_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss + self.reg_weight * reg_loss

    def full_predict(self, users):
        if self.storage_final_user_embeddings is None:
            self.eval()
            with torch.no_grad():
                self.hierarchical_preference_propagation(return_intermediate=False)

        target_idx = self.n_behaviors - 1

        user_embeds = self.storage_final_user_embeddings[users, target_idx]
        all_item_embeds = self.storage_final_item_embeddings[:, target_idx]

        scores = torch.matmul(user_embeds, all_item_embeds.t())
        return scores