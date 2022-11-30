import copy
import os
import time
import torch
from torch import nn
from absl import flags
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from os import path
import wandb
import logging

from torch_geometric.utils import negative_sampling

from .utils import get_time_bundle, write_results
from ..eval import eval_all
from ..scheduler import CosineDecayScheduler
from ..utils import compute_data_representations_only
from ..transforms import compose_transforms

from ..models.decoders import MLPProdDecoder

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

FLAGS = flags.FLAGS


#####
# Train & eval functions
#####
def train_epoch(model, predictor, optimizer, training_data, criterion):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    train_edge = training_data.edge_index
    # Perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(edge_index=train_edge,
                                       num_nodes=training_data.num_nodes,
                                       num_neg_samples=train_edge.size(1),
                                       method='sparse')

    edge_label_index = torch.cat(
        [train_edge, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([train_edge.new_ones(train_edge.size(1)),
                            train_edge.new_zeros(neg_edge_index.size(1))],
                           dim=0)

    model_out = model(training_data)
    edge_embeddings = model_out[edge_label_index]
    combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
    out = predictor(combined)

    loss = criterion(out.view(-1), edge_label.float())
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def eval_model(model, predictor, inference_data, device, eval_edge, eval_edge_neg):
    model.eval()
    predictor.eval()
    n_edges = eval_edge.size(1)

    edge_label_index = torch.cat(
        [eval_edge, eval_edge_neg],
        dim=-1,
    ).to(device)

    model_out = model(inference_data)
    embeddings = nn.Embedding.from_pretrained(model_out, freeze=True)
    edge_embeddings = embeddings(edge_label_index)
    combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
    out = predictor.predict(combined).view(-1)

    y_pred_pos, y_pred_neg = out[:n_edges], out[n_edges:]
    return eval_all(y_pred_pos, y_pred_neg)


def perform_e2e_transductive_training(model_name, data, edge_split, output_dir, representation_size, device, input_size,
                                  has_features: bool, g_zoo):
    model = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes,
                            n_feats=data.x.size(1)).to(device)
    predictor = MLPProdDecoder(representation_size, hidden_size=FLAGS.link_mlp_hidden_size).to(device)

    # optimizer
    optimizer = Adam(list(model.parameters()) + list(predictor.parameters()), lr=FLAGS.lr)
    criterion = BCEWithLogitsLoss()

    valid_edge, test_edge = edge_split['valid']['edge'].T.to(device), edge_split['test']['edge'].T.to(device)
    valid_edge_neg, test_edge_neg = edge_split['valid']['edge_neg'].T.to(device), edge_split['test']['edge_neg'].T.to(
        device)

    best_val = None
    best_results = None
    target_metric = 'hits@50'
    last_epoch = None

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train_epoch(model, predictor, optimizer, data, criterion)
        val_res = eval_model(model, predictor, data, device, valid_edge, valid_edge_neg)
        test_res = eval_model(model, predictor, data, device, test_edge, test_edge_neg)

        if epoch % 10 == 0:
            print('Validation:', val_res)
            print('Test:', test_res)
        metric_names = list(test_res.keys())

        assert metric_names is not None

        for metric_name in metric_names:
            val_hits = val_res[metric_name]
            test_hits = test_res[metric_name]
            if wandb is not None:
                wandb.log({
                    f'val_{metric_name}': val_hits,
                    f'test_{metric_name}': test_hits,
                    'epoch': epoch
                },
                          step=wandb.run.step)

        if best_val is None or val_res[target_metric] > best_val[target_metric]:
            best_val = val_res
            best_results = test_res
            last_epoch = epoch

            wandb.log(
                {
                    **{f'best_{metric_name}': best_results[metric_name] for metric_name in metric_names}, 'epoch': epoch
                },
                step=wandb.run.step)

        # Early stopping
        if last_epoch is not None and epoch - last_epoch > 50:
            break

    print('Best results: ', best_results)
    torch.save((model, predictor), os.path.join(output_dir, 'model.pt'))
    print('Saved model weights')

    all_results = [{
        'target_metric': target_metric,
        'type': 'prod',
        'val': best_val,
        'test': best_results,
        'fixed': True
    }]
    write_results(model_name, output_dir, all_results)

    return model, predictor, all_results




def perform_e2e_inductive_training(model_name, training_data, val_data, inference_data, data, test_edge_bundle,
                               negative_samples, output_dir, representation_size, device, input_size, has_features,
                               g_zoo):
    (test_old_old_ei, test_old_new_ei, test_new_new_ei, test_edge_index) = test_edge_bundle

    model = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes,
                            n_feats=data.x.size(1)).to(device)
    predictor = MLPProdDecoder(representation_size, hidden_size=FLAGS.link_mlp_hidden_size).to(device)

    # optimizer
    optimizer = Adam(list(model.parameters()) + list(predictor.parameters()), lr=FLAGS.lr)
    criterion = BCEWithLogitsLoss()

    # we already filtered out test/val edges
    training_data = training_data.to(device)
    inference_data = inference_data.to(device)

    best_val = None
    best_results = None
    target_metric = 'hits@50'
    last_epoch = 0

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        train_loss = train_epoch(model, predictor, optimizer, training_data, criterion)
        info = (model, predictor, inference_data, device)

        val_res = eval_model(*info, val_data.edge_label_index[:, val_data.edge_label == 1],
                             val_data.edge_label_index[:, val_data.edge_label == 0])
        test_res = eval_model(*info, test_edge_index, negative_samples)
        new_new_res = eval_model(*info, test_new_new_ei, negative_samples)
        old_new_res = eval_model(*info, test_old_new_ei, negative_samples)
        old_old_res = eval_model(*info, test_old_old_ei, negative_samples)
        metric_names = list(test_res.keys())

        if epoch % 5 == 0:
            print('Validation:', val_res)
            print('Test:', test_res)
            print('New-New:', new_new_res)
            print('Old-New:', old_new_res)
            print('Old-Old:', old_old_res)

        # Store best results based on validation hits@50
        if best_val is None or val_res[target_metric] > best_val[target_metric]:
            best_val_res = val_res
            best_test_res = test_res
            best_old_old_res = old_old_res
            best_old_new_res = old_new_res
            best_new_new_res = new_new_res

            wandb.log(
                {
                    **{f'best_{metric_name}': best_test_res[metric_name] for metric_name in metric_names},
                    **{f'best_old-old_{metric_name}': best_old_old_res[metric_name] for metric_name in metric_names},
                    **{f'best_old-new_{metric_name}': best_old_new_res[metric_name] for metric_name in metric_names},
                    **{f'best_new-new_{metric_name}': best_new_new_res[metric_name] for metric_name in metric_names},
                    'epoch': epoch  # yapf: disable
                },
                step=wandb.run.step)
        elif epoch - last_epoch > 50:
            break

        metric_names = list(test_res.keys())

        for metric_name in metric_names:
            wandb.log(
                {
                    f'val_{metric_name}': val_res[metric_name],
                    f'test_{metric_name}': test_res[metric_name],
                    f'oldold_{metric_name}': old_old_res[metric_name],
                    f'oldnew_{metric_name}': old_new_res[metric_name],
                    f'newnew_{metric_name}': new_new_res[metric_name],
                    'epoch': epoch
                },
                step=wandb.run.step)
        wandb.log({'train_loss': train_loss}, step=wandb.run.step)

    print('Best results: ', best_results)
    torch.save((model, predictor), path.join(output_dir, 'model.pt'))
    print('Saved model weights')

    results = {
        'target_metric': target_metric,
        'type': 'prod',
        'val': best_val_res,
        'test': best_test_res,
        'old_old': best_old_old_res,
        'old_new': best_old_new_res,
        'new_new': best_new_new_res
    }

    all_results = [results]
    write_results(model_name, output_dir, all_results)

    return model, predictor, all_results