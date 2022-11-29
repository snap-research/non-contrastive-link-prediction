import copy

import torch
from absl import flags

from .models import GraceEncoder

FLAGS = flags.FLAGS

from .utils import add_node_feats
from torch_geometric.loader import NeighborLoader


class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.
    From:
    https://github.com/nerdslab/bgrl/blob/dec99f8c605e3c4ae2ece57f3fa1d41f350d11a9/bgrl/bgrl.py

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """

    def __init__(self, encoder, predictor, has_features):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor
        self.has_features = has_features

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Link features together
        if not self.has_features:
            self.target_encoder.node_feats = self.online_encoder.node_feats

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y


def compute_representations_only(net, dataset, device, has_features=True, feature_type='degree'):
    r"""Pre-computes the representations for the entire dataset.
    Does not include node labels.

    Returns:
        torch.Tensor: Representations
    """
    net.eval()
    reps = []

    for data in dataset:
        # forward
        data = data.to(device)
        if not has_features:
            if data.x is not None:
                print('[WARNING] features overidden in adj matrix')
            data.x = net.get_node_feats().weight.data
        elif data.x is None:
            data = add_node_feats(data, device=device, type=feature_type)

        with torch.no_grad():
            if isinstance(net, GraceEncoder):
                reps.append(net(data.x, data.edge_index))
            else:
                reps.append(net(data))

    reps = torch.cat(reps, dim=0)
    return reps

def compute_data_representations_only(net, data, device, has_features=True):
    r"""Pre-computes the representations for the entire dataset.
    Does not include node labels.

    Returns:
        torch.Tensor: Representations
    """
    net.eval()
    reps = []

    if not has_features:
        if data.x is not None:
            print('[WARNING] features overidden in adj matrix')
        data.x = net.get_node_feats().weight.data

    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0).to(device)
    return reps


class TripletBGRL(torch.nn.Module):
    """Triplet-BGRL class.
    Similar to the BGRL class, but contains a forward_target function
    that allows passing additional data through the target network.
    """

    def __init__(self, encoder, predictor, has_features):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor
        self.has_features = has_features

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Link features together
        if not self.has_features:
            self.target_encoder.node_feats = self.online_encoder.node_feats

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    @torch.no_grad()
    def forward_target(self, target_x):
        return self.target_encoder(target_x).detach()

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y
