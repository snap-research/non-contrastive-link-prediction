import copy
from typing import List

import torch

from .utils import add_node_feats


class GraphBarlowTwins(torch.nn.Module):
    r"""Graph Barlow Twins implementation.
    """

    def __init__(self, encoder, has_features):
        super().__init__()
        # online network
        self.encoder = encoder
        self.has_features = has_features

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.encoder.parameters())

    def forward(self, online_x, target_x):
        rep_a = self.encoder(online_x)
        rep_b = self.encoder(target_x)
        return rep_a, rep_b


class CCASSG(torch.nn.Module):
    r"""CCA-SSG implementation.
    """

    def __init__(self, encoder, has_features):
        super().__init__()
        # online network
        self.encoder = encoder
        self.has_features = has_features

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.encoder.parameters())

    def forward(self, online_x, target_x):
        rep_a = self.encoder(online_x)
        rep_b = self.encoder(target_x)

        z1 = (rep_a - rep_a.mean(0)) / rep_a.std(0)
        z2 = (rep_b - rep_b.mean(0)) / rep_b.std(0)
        return z1, z2


class SimSiam(torch.nn.Module):

    def __init__(self, encoder, predictor, has_features):
        super().__init__()
        # online network
        self.encoder = encoder
        self.predictor = predictor
        self.has_features = has_features

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.encoder.parameters()) + list(self.predictor.parameters())

    def forward(self, online_x, target_x):
        rep_a = self.encoder(online_x)
        rep_a = self.predictor(rep_a)

        with torch.no_grad():
            rep_b = self.encoder(target_x)
        return rep_a, rep_b
