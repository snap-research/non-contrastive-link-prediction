import torch


class CcaSsg(torch.nn.Module):
    """CCA-SSG implementation.
    Very similar to the GBT implementation, but z-normalizes output vectors.
    """

    def __init__(self, encoder, has_features):
        super().__init__()
        self.encoder = encoder
        self.has_features = has_features

    def trainable_parameters(self):
        """Returns the parameters that will be updated via an optimizer."""
        return list(self.encoder.parameters())

    def forward(self, online_x, target_x):
        rep_a = self.encoder(online_x)
        rep_b = self.encoder(target_x)

        z1 = (rep_a - rep_a.mean(0)) / rep_a.std(0)
        z2 = (rep_b - rep_b.mean(0)) / rep_b.std(0)
        return z1, z2
