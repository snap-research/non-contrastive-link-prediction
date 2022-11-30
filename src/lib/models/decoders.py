from typing import List
import torch
from torch import nn


class LinkPredictorZoo:
    """Class that allows switching between different link prediction decoders.
    Two models are currently supported:
      - prod_mlp: a Hadamard product MLP
      - mlp: a concatenation-based MLP
    """

    def __init__(self, flags):
        self.models = {'mlp': MLPConcatDecoder, 'prod_mlp': MLPProdDecoder}
        self.flags = flags

    def init_model(self, model_class, embedding_size):
        flags = self.flags
        if model_class == MLPConcatDecoder:
            if flags.adjust_layer_sizes:
                return MLPConcatDecoder(embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size * 2)
            return MLPConcatDecoder(embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size)
        elif model_class == MLPProdDecoder:
            return MLPProdDecoder(embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size)

    def filter_models(self, models: List[str]):
        return [model for model in models if model in self.models]

    def check_model(self, model_name):
        """Checks if a model with the given name exists.
        Raises an error if not.
        """

        if model_name not in self.models:
            raise ValueError(f'Unknown predictor model "{model_name}"')

    def get_model(self, model_name, embedding_size):
        """Given a model name, return the corresponding model class."""
        self.check_model(model_name)
        return self.init_model(self.models[model_name], embedding_size)


class MLPConcatDecoder(torch.nn.Module):
    """Concatentation-based MLP link predictor.
    """

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class MLPProdDecoder(torch.nn.Module):
    """Hadamard-product-based MLP link predictor.
    """

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        # self.embeddings = embeddings
        self.net = nn.Sequential(nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x):
        left, right = x[:, :self.embedding_size], x[:, self.embedding_size:]
        return self.net(left * right)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))
