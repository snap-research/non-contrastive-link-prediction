from typing import List
import torch
from torch import nn


class LinkPredictorZoo:

    def __init__(self, flags):
        self.models = {'mlp': MLPLinkPredictor, 'prod_mlp': MLPProdLinkPredictor}
        self.flags = flags

    def init_model(self, model_class, embedding_size):
        flags = self.flags
        if model_class == MLPLinkPredictor:
            if flags.adjust_layer_sizes:
                return MLPLinkPredictor(embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size * 2)
            return MLPLinkPredictor(embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size)
        elif model_class == MLPProdLinkPredictor:
            return MLPProdLinkPredictor(embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size)

    def filter_models(self, models: List[str]):
        return [model for model in models if model in self.models]

    # Function to test if the model exists
    # Raise an error if not
    def check_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f'Unknown predictor model "{model_name}"')

    def get_model(self, model_name, embedding_size):
        self.check_model(model_name)
        return self.init_model(self.models[model_name], embedding_size)


class MLPLinkPredictor(torch.nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_size * 2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class MLPProdLinkPredictor(torch.nn.Module):

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
