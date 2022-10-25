import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-15


# From the official Graph Barlow Twins PyTorch implementation
def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = ((1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(2).sum())

    return loss

def cca_ssg_loss(z1, z2, cca_lambda, N):
    c = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)

    c = c / N
    c1 = c1 / N
    c2 = c2 / N
    loss_inv = -torch.diagonal(c).sum()
    iden = torch.eye(c.shape[0]).to(z1.device)
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()

    return (1 - cca_lambda) * loss_inv + cca_lambda * (loss_dec1 + loss_dec2)
