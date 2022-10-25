import copy
import math
import random
from typing import Callable, List, Tuple, Union

import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from torch_geometric.transforms import BaseTransform


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

VALID_BATCH_TRANSFORMS = dict({'standard': ['BatchDropEdges', 'BatchDropFeatures']})


class ChooserTransformation:

    def __init__(self, transformations, transformation_args):
        self.transformations = [transformations[i](*transformation_args[i]) for i in range(len(transformations))]
        self.transformations_str = ',\n'.join([str(x) for x in transformations])

    def __call__(self, data):
        transformation = random.choice(self.transformations)
        return transformation(data)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.transformations_str)


def compose_transforms(transform_name, drop_edge_p, drop_feat_p, mid_drop_p=0.1, create_copy=True, use_batch=False):
    if use_batch:
        return compose_batch_transform(transform_name,
                                       drop_edge_p=drop_edge_p,
                                       drop_feat_p=drop_feat_p,
                                       mid_drop_p=mid_drop_p,
                                       create_copy=create_copy)

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
    # copy graph
    if create_copy:
        transforms.append(copy.deepcopy)

    for transform_name in catalog:
        transform_class, transform_feats = feats[transform_name]
        transforms.append(transform_class(*transform_feats))

    return Compose(transforms)


###
# Next 3 functions are from
# # from https://github.com/pbielak/graph-barlow-twins/blob/ec62580aa89bf3f0d20c92e7549031deedc105ab/gssl/augment.py
###
def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))


def mask_features(x: torch.Tensor, p: float) -> torch.Tensor:
    num_features = x.size(-1)
    device = x.device

    return bernoulli_mask(size=(1, num_features), prob=p).to(device) * x


def drop_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    num_edges = edge_index.size(-1)
    device = edge_index.device

    mask = bernoulli_mask(size=num_edges, prob=p).to(device) == 1.

    return edge_index[:, mask]


###


# batch transforms take 2 inputs
# x: the feature matrix
# adjs: the list of Edge indexes
class BatchMaskFeatures:

    def __init__(self, p=None):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, x: torch.Tensor, adjs: Union[List[EdgeIndex], List[torch.Tensor]]):
        return mask_features(x, self.p), adjs

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class BatchDropEdges:

    def __init__(self, p=None):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, x: torch.Tensor, adjs: List[EdgeIndex]):
        return x, [drop_edges(adj.edge_index, self.p) for adj in adjs]

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class BatchCompose(BaseTransform):
    """Composes several transforms together.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor, adjs: List[EdgeIndex]):
        for transform in self.transforms:
            x, adjs = transform(x, adjs)
        return x, adjs

    def __repr__(self) -> str:
        args = [f'  {transform}' for transform in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, ',\n'.join(args))


def compose_batch_transform(transform_name, drop_edge_p, drop_feat_p, mid_drop_p=0.1, create_copy=True):
    if transform_name not in VALID_BATCH_TRANSFORMS:
        raise ValueError(f'{transform_name} is not a valid batch transformation.')

    feats = {'BatchDropEdges': (BatchDropEdges, [drop_edge_p]), 'BatchDropFeatures': (BatchMaskFeatures, [drop_feat_p])}

    catalog = VALID_BATCH_TRANSFORMS[transform_name]

    transforms = []
    for transform_name in catalog:
        transform_class, transform_feats = feats[transform_name]
        transforms.append(transform_class(*transform_feats))

    return BatchCompose(transforms)
