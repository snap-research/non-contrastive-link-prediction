from torch import nn


class MLP_Predictor(nn.Module):
    r"""MLP used for the BGRL/T-BGRL predictor. The MLP has one hidden layer.
    This function is from https://github.com/nerdslab/bgrl/blob/dec99f8c605e3c4ae2ece57f3fa1d41f350d11a9/bgrl/predictors.py

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """

    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_size, hidden_size, bias=True), nn.PReLU(1),
                                 nn.Linear(hidden_size, output_size, bias=True))
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
