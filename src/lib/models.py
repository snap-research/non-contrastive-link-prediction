import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, SAGEConv, Sequential, GINConv, GATConv
import torch.nn.functional as F
from torch_geometric.data import Data


class EncoderZoo:

    def __init__(self, flags):
        self.models = {'gcn': GCN, 'sage': NormalSAGE, 'gat': GAT, 'gin': GIN, 'std-sage': SAGE}
        self.flags = flags

    def _init_model(self, model_class, input_size: int, use_feat: bool, n_nodes: int, batched: bool, n_feats: int):
        flags = self.flags
        if model_class == GCN:
            # TODO(author): make this a param?
            return GCN([input_size] + flags.graph_encoder_layer,
                       batchnorm=True,
                       use_feat=use_feat,
                       n_nodes=n_nodes,
                       batched=batched)
        elif model_class == NormalSAGE:
            return NormalSAGE([input_size] + flags.graph_encoder_layer,
                              batchnorm=True,
                              use_feat=use_feat,
                              n_nodes=n_nodes,
                              batched=batched)
        elif model_class == GAT:
            return GAT([input_size] + flags.graph_encoder_layer,
                       batchnorm=True,
                       use_feat=use_feat,
                       n_nodes=n_nodes,
                       batched=batched)
        elif model_class == GIN:
            return GIN([input_size] + flags.graph_encoder_layer,
                       batchnorm=True,
                       use_feat=use_feat,
                       n_nodes=n_nodes,
                       batched=batched,
                       n_feats=n_feats)
        elif model_class == SAGE:
            return SAGE(input_size, 256, 256, 2, dropout=0.5)
        elif model_class == GraphSAGE_GCN:
            if not use_feat:
                raise NotImplementedError('Featureless GraphSAGE not yet implemented')
            if len(flags.graph_encoder_layer) > 2:
                raise ValueError('Too many layers for GraphSAGE - only 2 layers currently supported')

            s1, s2 = flags.graph_encoder_layer
            return GraphSAGE_GCN(input_size, s1, s2, batched=batched)

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


class SAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        adj_t = data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class NormalSAGE(nn.Module):

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
                layers.append(SAGEConv(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append((SAGEConv(in_dim, out_dim), 'x, edge_index -> x'),)

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

    def partial_forward(self, x, adjs):
        raise NotImplementedError('Todo: revisit')

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


class GAT(GCN):

    def __init__(self,
                 layer_sizes,
                 batchnorm=False,
                 batchnorm_mm=0.99,
                 layernorm=False,
                 weight_standardization=False,
                 use_feat=True,
                 n_nodes=0,
                 batched=False):
        super(GCN, self).__init__()
        # super().__init__(layer_sizes, batchnorm=batchnorm, batchnorm_mm=batchnorm_mm, layernorm=layernorm, weight_standardization=weight_standardization, use_feat=use_feat, n_nodes=n_nodes, batched=batched)

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
                layers.append(GATConv(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append((GATConv(in_dim, out_dim), 'x, edge_index -> x'),)

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


class GIN(GCN):

    def __init__(self,
                 layer_sizes,
                 batchnorm=False,
                 batchnorm_mm=0.99,
                 layernorm=False,
                 weight_standardization=False,
                 use_feat=True,
                 n_nodes=0,
                 batched=False,
                 n_feats=None):
        super(GCN, self).__init__()
        # super(GCN, self).__init__(layer_sizes, batchnorm=batchnorm, batchnorm_mm=batchnorm_mm, layernorm=layernorm, weight_standardization=weight_standardization, use_feat=use_feat, n_nodes=n_nodes, batched=batched)

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.n_layers = len(layer_sizes)
        self.batched = batched
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        relus = []
        batchnorms = []
        mlp_layers = []

        for _, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            mlp_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.PReLU(),
                    nn.Linear(out_dim, out_dim),
                    nn.PReLU(),
                    nn.BatchNorm1d(out_dim),
                ))
            if batched:
                layers.append(GINConv(mlp_layers[-1], in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append((GINConv(mlp_layers[-1], in_dim, out_dim), 'x, edge_index -> x'),)

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
        self.mlp_layers = nn.ModuleList(mlp_layers)

        self.use_feat = use_feat
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[1])

class GraphSAGE_GCN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, batched=False):
        super().__init__()
        self.batched = batched
        # TODO(author): consider adding BatchNorm?

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, embedding_size, root_weight=True),
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
        ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(1),
            nn.PReLU(1),
            nn.PReLU(1),
        ])

    def partial_forward(self, x, adjs):
        h1 = self.convs[0](x, adjs[0].edge_index)
        h1 = self.layer_norms[0](h1)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](h1 + x_skip_1, adjs[1].edge_index)
        h2 = self.layer_norms[1](h2)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](h1 + h2 + x_skip_2, adjs[2].edge_index)
        ret = self.layer_norms[2](ret)
        ret = self.activations[2](ret)
        return ret
        # for i, (edge_index, _, size) in enumerate(adjs):
        #     x_target = x[:size[1]]
        #     x = self.convs[i]((x, x_target), edge_index)
        #     x = self.activations[i](x)
        #     x = self.batchnorms[i](x)
        # return x

    def split_forward(self, x, edge_index):
        return self.forward(Data(x, edge_index))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        h1 = self.convs[0](x, edge_index)
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](h1 + x_skip_1, edge_index)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](h1 + h2 + x_skip_2, edge_index)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()

    # TODO(author): maybe don't hardcode this?
    @property
    def num_layers(self):
        return 3
