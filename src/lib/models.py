import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential
import torch.nn.functional as F


class EncoderZoo:
    """Returns an encoder of the specified type.
    """

    def __init__(self, flags):
        self.models = { 'gcn': GCN }
        self.flags = flags

    def _init_model(self, model_class, input_size: int, use_feat: bool, n_nodes: int, batched: bool, n_feats: int):
        flags = self.flags
        if model_class == GCN:
            return GCN([input_size] + flags.graph_encoder_layer,
                       batchnorm=True,
                       use_feat=use_feat,
                       n_nodes=n_nodes,
                       batched=batched)

    # Function to test if the model exists
    # Raise an error if not
    def check_model(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(f'Unknown predictor model "{model_name}"')

    def get_model(self,
                  model_name: str,
                  input_size: int,
                  use_feat: bool,
                  n_nodes: int,
                  n_feats: int,
                  batched: bool = False):
        self.check_model(model_name)
        return self._init_model(self.models[model_name],
                                input_size,
                                use_feat,
                                n_nodes,
                                batched=batched,
                                n_feats=n_feats)


class GCN(nn.Module):
    """Basic GCN encoder.
    This is based off of the official BGRL encoder implementation.
    """

    def __init__(self,
                 layer_sizes,
                 batchnorm=False,
                 batchnorm_mm=0.99,
                 layernorm=False,
                 weight_standardization=False,
                 use_feat=True,
                 n_nodes=0,
                 batched=False):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.n_layers = len(layer_sizes)
        self.batched = batched
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        relus = []
        batchnorms = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if batched:
                layers.append(GCNConv(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)

                if batchnorm:
                    layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
                else:
                    layers.append(LayerNorm(out_dim))

                layers.append(nn.PReLU())

        if batched:
            self.convs = nn.ModuleList(layers)
            self.relus = nn.ModuleList(relus)
            self.batchnorms = nn.ModuleList(batchnorms)
        else:
            self.model = Sequential('x, edge_index', layers)

        self.use_feat = use_feat
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[1])

    def split_forward(self, x, edge_index):
        if self.weight_standardization:
            self.standardize_weights()
        if self.use_feat:
            return self.model(x, edge_index)
        return self.model(self.node_feats.weight.data.clone(), edge_index)

    def forward(self, data):
        if not self.batched:
            if self.weight_standardization:
                self.standardize_weights()
            if self.use_feat:
                return self.model(data.x, data.edge_index)
            return self.model(self.node_feats.weight.data.clone(), data.edge_index)
        # otherwise, batched
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            x = self.relus[i](x)
            x = self.batchnorms[i](x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

    def get_node_feats(self):
        if hasattr(self, 'node_feats'):
            return self.node_feats
        return None

    @property
    def num_layers(self):
        return self.n_layers

class GraceEncoder(torch.nn.Module):
    """Encoder used by GRACE.
    From: https://github.com/CRIPAC-DIG/GRACE/blob/51b44961b68b2f38c60f85cf83db13bed8fd0780/model.py

    Args:
        torch (_type_): _description_
    """
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2):
        super(GraceEncoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k - 1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x

class GraceModel(torch.nn.Module):

    def __init__(self, encoder: GraceEncoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GraceModel, self).__init__()
        self.encoder: GraceEncoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag() /
                                     (refl_sim.sum(1) + between_sim.sum(1) - refl_sim[:, i * batch_size:
                                                                                      (i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
