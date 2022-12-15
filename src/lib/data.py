from typing import Union
import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset, HeteroData, Data
from torch_geometric.transforms import BaseTransform, Compose, NormalizeFeatures
from torch_geometric.utils import to_undirected
from ogb.linkproppred import PygLinkPropPredDataset


class ConvertToFloat(BaseTransform):
    """Tranform to convert features to floats."""

    def __call__(self, data: Union[Data, HeteroData]):
        if data.x is not None:
            data.x = data.x.float()  # type: ignore
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def get_dataset(root, name, transform=Compose([ConvertToFloat(), NormalizeFeatures()])):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=name, root=root, transform=transform)
        return dataset

    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
        'cora': (datasets.Planetoid, 'Cora'),
        'citeseer': (datasets.Planetoid, 'Citeseer'),
        'crocodile': (datasets.WikipediaNetwork, 'crocodile'),
        'squirrel': (datasets.WikipediaNetwork, 'squirrel'),
        'chameleon': (datasets.WikipediaNetwork, 'chameleon'),
        'texas': (datasets.WebKB, 'Texas'),
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(
        list(pyg_dataset_dict.keys())
    )

    dataset_class, name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=name, transform=transform)

    return dataset


class PygConcatDataset(InMemoryDataset):
    """PyG Dataset class for merging multiple Dataset objects into one."""

    def __init__(self, datasets):
        super(PygConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)
