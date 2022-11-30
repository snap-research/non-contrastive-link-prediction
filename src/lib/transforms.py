import copy
import math
import random

import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.dropout import dropout_adj


class DropFeatures:
    r"""Drops node features with probability p."""

    def __init__(self, p=None):
        assert p is not None
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class ScrambleFeatures:
    r"""Randomly scrambles the rows of the feature matrix."""

    def __call__(self, data):
        row_perm = torch.randperm(data.x.size(0))
        data.x = data.x[row_perm, :]
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class RandomEdges:
    r"""Completely randomize the edge index"""

    def __call__(self, data):
        n = data.num_nodes
        data.edge_index = torch.randint_like(data.edge_index, n - 1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RandomRangeEdges:
    r"""Completely randomize the edge index"""

    def __call__(self, data):
        n = data.num_nodes
        n_edges = data.edge_index.size(1)

        n_edges = random.randint(math.ceil(n_edges * 0.75), math.ceil(n_edges * 1.25))
        data.edge_index = torch.randint(0, n - 1, (2, n_edges), dtype=data.edge_index.dtype).to(data.edge_index.device)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class DropEdges:
    r"""Drops edges with probability p."""

    def __init__(self, p, force_undirected=False):
        assert p is not None
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)


class AddEdges:
    """Perform random edge addition."""

    def __init__(self, sample_size_ratio=0.1):
        self.sample_size_ratio = sample_size_ratio

    def __call__(self, data):
        edge_index = data.edge_index
        n_samples = round(self.sample_size_ratio * edge_index)
        neg_edges = negative_sampling(data.edge_index, num_nodes=data.num_nodes, num_neg_samples=n_samples)

        edge_index = torch.cat((edge_index, neg_edges))
        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}(sample_size_ratio={})'.format(self.__class__.__name__, self.sample_size_ratio)


class RandomizeFeatures:
    """Completely randomize the feature matrix (maintain the same size)."""

    def __init__(self):
        pass

    def __call__(self, data):
        data.x = torch.rand_like(data.x)
        return data

    def __repr__(self):
        return '{}(sample_size_ratio={})'.format(self.__class__.__name__, self.sample_size_ratio)


VALID_TRANSFORMS = dict({
    'standard': ['DropEdges', 'DropFeatures'],
    'all': ['DropEdges', 'DropFeatures'],
    'none': [],
    'drop-edge': ['DropEdges'],
    'drop-feat': ['DropFeatures'],
    'add-edges': ['AddEdges'],
    'add-edges-feat-drop': ['AddEdges', 'DropFeatures']
})

VALID_NEG_TRANSFORMS = dict({
    'heavy-sparsify': ['DropEdges', 'DropFeatures'],
    'randomize-feats': ['RandomizeFeatures'],
    'scramble-feats': ['ScrambleFeatures'],
    'randomize-drop-combo': ['DropEdges', 'RandomizeFeatures'],
    'scramble-drop-combo': ['ScrambleFeatures', 'DropEdges'],
    'scramble-edge-combo': ['ScrambleFeatures', 'RandomEdges'],
    'rand-rand-combo': ['RandomizeFeatures', 'RandomEdges'],
    'rand-rand-rand-combo': ['RandomizeFeatures', 'RandomRangeEdges'],
    'scramble-edge-choice': ['ScrambleFeaturesOrRandomEdges'],
    'scramble-drop-choice': ['ScrambleFeatOrDropEdges'],
    'random-edges': ['RandomEdges'],
    'all-choice': ['AllChoice']
})


class ChooserTransformation:
    """Consists of multiple transformations.
    When this transform is called, each of those transforms is selected and called with
    uniform probability. This allows for alternating transforms during model training.
    """

    def __init__(self, transformations, transformation_args):
        self.transformations = [transformations[i](*transformation_args[i]) for i in range(len(transformations))]
        self.transformations_str = ',\n'.join([str(x) for x in transformations])

    def __call__(self, data):
        transformation = random.choice(self.transformations)
        return transformation(data)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.transformations_str)


def compose_transforms(transform_name, drop_edge_p, drop_feat_p, create_copy=True):
    """Given a flag-friendly transform name, returns the corresponding transformation object.
    Note that transforms include both augmentations and corruptions.
    The dictionary of valid augmentations can be found in `VALID_TRANSFORMS`.
    The dictionary of valid corruptions can be foudn in `VALID_NEG_TRANSFORMS`.
    """

    if transform_name in VALID_TRANSFORMS:
        catalog = VALID_TRANSFORMS[transform_name]
    elif transform_name in VALID_NEG_TRANSFORMS:
        catalog = VALID_NEG_TRANSFORMS[transform_name]
    else:
        raise ValueError('Unknown transform_name: ', transform_name)

    # Create matching of transforms -> features
    feats = {
        'DropEdges': (DropEdges, [drop_edge_p]),
        'DropFeatures': (DropFeatures, [drop_feat_p]),
        'AddEdges': (AddEdges, []),
        'RandomizeFeatures': (RandomizeFeatures, []),
        'ScrambleFeatures': (ScrambleFeatures, []),
        'RandomEdges': (RandomEdges, []),
        'RandomRangeEdges': (RandomRangeEdges, []),
        'ScrambleFeaturesOrRandomEdges': (ChooserTransformation, [(ScrambleFeatures, RandomEdges), ([], [])]),
        'ScrambleFeatOrDropEdges': (ChooserTransformation, [(ScrambleFeatures, DropEdges), ([], [0.95])]),
        'AllChoice':
            (ChooserTransformation, [(ScrambleFeatures, RandomEdges, RandomizeFeatures, DropFeatures, DropEdges),
                                     ([], [], [], [0.95], [0.95])])
    }

    transforms = []
    if create_copy:
        transforms.append(copy.deepcopy)

    for transform_name in catalog:
        transform_class, transform_feats = feats[transform_name]
        transforms.append(transform_class(*transform_feats))

    return Compose(transforms)
