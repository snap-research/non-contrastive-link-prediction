from typing import List
import torch
from torch import nn
from enum import Enum


class DecoderModel(Enum):
    CONCAT_MLP = 'concat_mlp'
    PRODUCT_MLP = 'prod_mlp'


class MlpConcatDecoder(torch.nn.Module):
    """Concatentation-based MLP link predictor."""

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class MlpProdDecoder(torch.nn.Module):
    """Hadamard-product-based MLP link predictor."""

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        # self.embeddings = embeddings
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        left, right = x[:, : self.embedding_size], x[:, self.embedding_size :]
        return self.net(left * right)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class DecoderZoo:
    """Class that allows switching between different link prediction decoders.
    Two models are currently supported:
      - prod_mlp: a Hadamard product MLP
      - mlp: a concatenation-based MLP

    Reads flags from an instance of absl.FlagValues.
    See ../lib/flags.py for flag defaults and descriptions.
    """

    # Note: we use the value of the enums since we read them in as flags
    models = {
        DecoderModel.CONCAT_MLP.value: MlpConcatDecoder,
        DecoderModel.PRODUCT_MLP.value: MlpProdDecoder,
    }

    def __init__(self, flags):
        self.flags = flags

    def init_model(self, model_class, embedding_size):
        flags = self.flags
        if model_class == MlpConcatDecoder:
            if flags.adjust_layer_sizes:
                return MlpConcatDecoder(
                    embedding_size=embedding_size,
                    hidden_size=flags.link_mlp_hidden_size * 2,
                )
            return MlpConcatDecoder(
                embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size
            )
        elif model_class == MlpProdDecoder:
            return MlpProdDecoder(
                embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size
            )

    @staticmethod
    def filter_models(models: List[str]):
        return [model for model in models if model in DecoderZoo.models]

    def check_model(self, model_name):
        """Checks if a model with the given name exists.
        Raises an error if not.
        """
        if model_name not in self.models:
            raise ValueError(f'Unknown decoder model: "{model_name}"')
        return True

    def get_model(self, model_name, embedding_size):
        """Given a model name, return the corresponding model class."""
        self.check_model(model_name)
        return self.init_model(self.models[model_name], embedding_size)
