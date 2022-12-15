import copy
import torch


class BGRL(torch.nn.Module):
    """BGRL architecture for Graph Representation Learning.
    From:
    https://github.com/nerdslab/bgrl/blob/dec99f8c605e3c4ae2ece57f3fa1d41f350d11a9/bgrl/bgrl.py

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    Note:
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
        return list(self.online_encoder.parameters()) + list(
            self.predictor.parameters()
        )

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, (
            "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        )
        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1.0 - mm)

    def forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y
