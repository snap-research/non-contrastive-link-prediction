import logging
import torch
from torch_geometric.utils import to_networkx
from torch.nn.functional import one_hot
from absl import flags
import pandas as pd

from .models import GraceEncoder

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
SMALL_DATASETS = set(['cora', 'citeseer'])
# Used for formatting output
SHORT_DIVIDER = '=' * 10
LONG_DIVIDER_STR = '=' * 30


def print_run_num(run_num):
    log.info(LONG_DIVIDER_STR)
    log.info(LONG_DIVIDER_STR)
    log.info(SHORT_DIVIDER + f'  Run #{run_num}  ' + SHORT_DIVIDER)
    log.info(LONG_DIVIDER_STR)
    log.info(LONG_DIVIDER_STR)


def add_node_feats(data, device, type='degree'):
    assert type == 'degree'

    G = to_networkx(data)
    degrees = torch.tensor([v for (_, v) in G.degree()])  # type: ignore
    data.x = one_hot(degrees).to(device).float()
    return data


def keywise_agg(dicts):
    df = pd.DataFrame(dicts)
    mean_dict = df.mean().to_dict()
    std_dict = df.std().to_dict()
    return mean_dict, std_dict


def keywise_prepend(d, prefix):
    out = {}
    for k, v in d.items():
        out[prefix + k] = v
    return out


def is_small_dset(dset):
    return dset in SMALL_DATASETS


def merge_multirun_results(all_results):
    """Merges results from multiple runs into a single dictionary."""
    runs = zip(*all_results)
    agg_results = []
    val_mean = test_mean = None

    for run_group in runs:
        group_type = run_group[0]['type']
        val_res = [run['val'] for run in run_group]
        test_res = [run['test'] for run in run_group]

        val_mean, val_std = keywise_agg(val_res)
        test_mean, test_std = keywise_agg(test_res)
        agg_results.append(
            {
                'type': group_type,
                'val_mean': val_mean,
                'val_std': val_std,
                'test_mean': test_mean,
                'test_std': test_std,
            }
        )

    assert val_mean is not None
    assert test_mean is not None
    return agg_results, {
        **keywise_prepend(val_mean, 'val_mean_'),
        **keywise_prepend(test_mean, 'test_mean_'),
    }


def compute_representations_only(
    net, dataset, device, has_features=True, feature_type='degree'
):
    """Pre-computes the representations for the entire dataset.
    Does not include node labels.

    Returns:
        torch.Tensor: Representations
    """
    net.eval()
    reps = []

    for data in dataset:
        # forward
        data = data.to(device)
        if not has_features:
            if data.x is not None:
                log.warn('[WARNING] node features overidden in Data object')
            data.x = net.get_node_feats().weight.data
        elif data.x is None:
            data = add_node_feats(data, device=device, type=feature_type)

        with torch.no_grad():
            if isinstance(net, GraceEncoder):
                reps.append(net(data.x, data.edge_index))
            else:
                reps.append(net(data))

    reps = torch.cat(reps, dim=0)
    return reps


def compute_data_representations_only(net, data, device, has_features=True):
    r"""Pre-computes the representations for the entire dataset.
    Does not include node labels.

    Returns:
        torch.Tensor: Representations
    """
    net.eval()
    reps = []

    if not has_features:
        if data.x is not None:
            log.warn('[WARNING] features overidden in adj matrix')
        data.x = net.get_node_feats().weight.data

    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0).to(device)
    return reps
