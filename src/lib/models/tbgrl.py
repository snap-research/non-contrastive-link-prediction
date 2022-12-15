import copy
import torch


class TripletBgrl(torch.nn.Module):
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

    @torch.no_grad()
    def forward_target(self, target_x):
        """Performs inference on the target encoder without autograd information."""
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
