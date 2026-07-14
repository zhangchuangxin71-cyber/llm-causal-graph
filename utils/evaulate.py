import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F


def compute_loss_para(adj, device):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    # Keep a scalar pos_weight to avoid an extra n×n weight tensor (~2GB on Yelp).
    weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def compute_vgae_loss(logits, adj, norm, vgae_model, weight_tensor):
    # Row-chunked weighted BCE to avoid multi-GiB temporaries on large graphs.
    eps = 1e-7
    n = logits.size(0)
    chunk = 1024
    total = logits.new_zeros(())
    numel = 0
    for i in range(0, n, chunk):
        rows = slice(i, min(i + chunk, n))
        probs = logits[rows].clamp(eps, 1.0 - eps)
        target = adj[rows]
        pos_term = target * (-torch.log(probs))
        neg_term = (1.0 - target) * (-torch.log(1.0 - probs))
        total = total + (weight_tensor * pos_term + neg_term).sum()
        numel += probs.numel()
    vgae_loss = norm * (total / numel)
    kl_divergence = 0.5 / logits.size(0) * (
            1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
        1).mean()
    vgae_loss = vgae_loss - kl_divergence
    return vgae_loss


def adjust_loss(elbo, vgae_loss, infer_loss, reweight=True):
    if reweight:
        loss = 0.1 * (elbo + vgae_loss) + 0.01 * infer_loss
    else:
        loss = elbo + 0.1 * vgae_loss
    return loss

