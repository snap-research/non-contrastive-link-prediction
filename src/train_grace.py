"""This file trains the GRACE model and evaluates it for link prediction.
This file contains pieces of code from the official GRACE implementation
at https://github.com/CRIPAC-DIG/GRACE
"""
import json
import os
import time
import wandb

from absl import app
from absl import flags
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from lib.utils import compute_representations_only, print_run_num
from lib.data import get_dataset
from ogb.linkproppred import PygLinkPropPredDataset

from lib.eval import do_all_eval, do_inductive_eval
from lib.models import GraceEncoder, GraceModel, DecoderZoo
from lib.training import get_time_bundle
import lib.flags as FlagHelper

from lib.split import do_transductive_edge_split, do_node_inductive_edge_split
from lib.utils import (
    is_small_dset,
    merge_multirun_results
)

######
# Flags
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FLAGS = flags.FLAGS

# Shared flags
FlagHelper.define_flags(FlagHelper.ModelGroup.GRACE)

# GRACE-specific flags
flags.DEFINE_enum(
    'activation_type', 'prelu', ['prelu', 'relu'], 'Which activation type to use'
)
flags.DEFINE_float('tau', 0.0, 'GRACE parameter')


def drop_feature(x, drop_prob):
    """GRACE feature dropping function.
    From: https://github.com/CRIPAC-DIG/GRACE/blob/51b44961b68b2f38c60f85cf83db13bed8fd0780/model.py#L120
    """
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def train_grace(
    model: GraceModel,
    optimizer,
    x,
    edge_index,
    drop_edge_rate_1,
    drop_edge_rate_2,
    drop_feature_rate_1,
    drop_feature_rate_2,
):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


######
# Main
######
def main(_):
    model_prefix = ''
    if FLAGS.model_name_prefix:
        model_prefix = f'{FLAGS.model_name_prefix}_'
    model_name = f'{model_prefix}GRACE_{FLAGS.dataset}'
    assert (
        FLAGS.drop_edge_p_1 != 0
        and FLAGS.drop_edge_p_2 != 0
        and FLAGS.drop_feat_p_1 != 0
        and FLAGS.drop_feat_p_2 != 0
        and FLAGS.tau != 0
    )

    wandb.init(
        project='ind-grace',
        config={'model_name': model_name, **FLAGS.flag_values_dict()},
    )
    if wandb.run is None:
        raise ValueError('Failed to initialize wandb run!')

    OUTPUT_DIR = os.path.join('./runs', FLAGS.dataset, f'{model_name}_{wandb.run.id}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    learning_rate = FLAGS.lr
    num_hidden = 256
    num_proj_hidden = 256
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[FLAGS.activation_type]
    base_model = ({'gcn': GCNConv})[FLAGS.graph_encoder_model]
    num_layers = 2

    drop_edge_rate_1 = FLAGS.drop_edge_p_1
    drop_edge_rate_2 = FLAGS.drop_edge_p_2
    drop_feature_rate_1 = FLAGS.drop_feat_p_1
    drop_feature_rate_2 = FLAGS.drop_feat_p_2
    tau = FLAGS.tau
    num_epochs = FLAGS.epochs
    weight_decay = FLAGS.weight_decay

    dataset = get_dataset('./data', FLAGS.dataset)
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if FLAGS.split_method == 'inductive':
        if isinstance(dataset, PygLinkPropPredDataset):
            raise NotImplementedError()

        (
            training_data,
            val_data,
            inference_data,
            data,
            test_edge_bundle,
            negative_samples,
        ) = do_node_inductive_edge_split(
            dataset=dataset,
            split_seed=FLAGS.split_seed,
            small_dataset=is_small_dset(FLAGS.dataset),
        )  # type: ignore
        training_data = training_data.to(device)
    else:  # transductive
        if isinstance(dataset, PygLinkPropPredDataset):
            # TODO(author): move it lower once we're sure this works properly
            edge_split = dataset.get_edge_split()
        else:
            edge_split = do_transductive_edge_split(dataset, FLAGS.split_seed)
            data.edge_index = edge_split['train']['edge'].t()  # type: ignore
        data.to(device)

    all_results = []
    all_times = []
    total_times = []

    for run_num in range(FLAGS.num_runs):
        print_run_num(run_num)

        dec_zoo = DecoderZoo(FLAGS)
        valid_models = DecoderZoo.filter_models(FLAGS.link_pred_model)

        encoder = GraceEncoder(
            dataset.num_features,
            num_hidden,
            activation,
            base_model=base_model,
            k=num_layers,
        ).to(device)
        model = GraceModel(encoder, num_hidden, num_proj_hidden, tau).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        times = []
        for epoch in range(1, num_epochs + 1):
            st_time = time.time_ns()
            if FLAGS.split_method == 'inductive':
                train_x = training_data.x
                train_ei = training_data.edge_index
            else:  # transductive
                train_x = data.x
                train_ei = data.edge_index

            loss = train_grace(
                model=model,
                optimizer=optimizer,
                x=train_x,
                edge_index=train_ei,
                drop_edge_rate_1=drop_edge_rate_1,
                drop_edge_rate_2=drop_edge_rate_2,
                drop_feature_rate_1=drop_feature_rate_1,
                drop_feature_rate_2=drop_feature_rate_2,
            )
            elapsed = time.time_ns() - st_time
            times.append(elapsed)

            if epoch % 25 == 0:
                log.debug(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        time_bundle = get_time_bundle(times)
        (total_time, std_time, mean_time, times) = time_bundle
        all_times.append(times.tolist())
        total_times.append(int(total_time))

        # incremental updates
        with open(f'{OUTPUT_DIR}/times.json', 'w') as f:
            json.dump({'all_times': all_times, 'total_times': total_times}, f)

        log.info("=== Final Evaluation ===")
        if FLAGS.split_method == 'inductive':
            results = do_inductive_eval(
                model_name=model_name,
                output_dir=OUTPUT_DIR,
                encoder=encoder,
                valid_models=valid_models,
                train_data=training_data,
                val_data=val_data,
                inference_data=inference_data,
                lp_zoo=dec_zoo,
                device=device,
                test_edge_bundle=test_edge_bundle,
                negative_samples=negative_samples,
                wb=wandb,
            )
        else:  # transductive
            representations = compute_representations_only(encoder, dataset, device)
            embeddings = torch.nn.Embedding.from_pretrained(representations)
            results, _ = do_all_eval(
                model_name,
                output_dir=OUTPUT_DIR,
                valid_models=valid_models,
                dataset=dataset,
                edge_split=edge_split,
                embeddings=embeddings,
                lp_zoo=dec_zoo,
                wb=wandb,
            )
        all_results.append(results)

    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)

    log.info(f'Done! Run information can be found at {OUTPUT_DIR}')


if __name__ == "__main__":
    app.run(main)
