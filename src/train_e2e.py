import logging
import os
from os import path
import json
import time

from absl import app
from absl import flags
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch_geometric.utils import negative_sampling
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import wandb

from lib.data import get_dataset
from lib.eval import eval_all
from lib.link_predictors import MLPProdDecoder
from lib.models import EncoderZoo
from lib.transforms import VALID_TRANSFORMS
from ogb.linkproppred import PygLinkPropPredDataset

from lib.utils import do_transductive_edge_split, do_node_inductive_edge_split, merge_multirun_results, set_random_seeds, write_results

######
# Flags
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FLAGS = flags.FLAGS

# Seeding
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('split_seed', 1, 'Split seed used to generate train/val/test split.')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'cora', 'citeseer'],
                  'Which graph dataset to use.')
flags.DEFINE_enum('split_method', 'transductive', ['inductive', 'transductive'],
                  'Which method to use to split the dataset (inductive or transductive).')
flags.DEFINE_enum('link_pred_model', 'mlp', ['mlp', 'prod_mlp', 'dot'], 'Which link prediction model to use')
flags.DEFINE_enum('graph_encoder_model', 'gcn', ['gcn', 'sage', 'std-sage'], 'Which graph encoder model to use')
flags.DEFINE_enum('graph_transforms', 'standard', list(VALID_TRANSFORMS.keys()), 'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', [256, 128], 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')
flags.DEFINE_integer('num_runs', 5, 'Number of times to train/evaluate the model and re-run')
flags.DEFINE_string('model_name_prefix', '', 'Prefix to prepend in front of the model name.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-3, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Link prediction model-specific flags
# MLP:
flags.DEFINE_integer('link_mlp_hidden_size', 128, 'Size of hidden layer in MLP for evaluation')
flags.DEFINE_float('link_mlp_lr', 0.01, 'Size of hidden layer in MLP for evaluation')
flags.DEFINE_integer('link_nn_epochs', 10000, 'Number of epochs in the NN for evaluation')
flags.DEFINE_enum('trivial_neg_sampling', 'false', ['true', 'false'], 'Whether or not to do trivial random sampling.')
flags.DEFINE_float('big_split_ratio', 0.2, 'Split ratio to use for larger datasets')


def get_full_model_name():
    model_prefix = ''
    if FLAGS.model_name_prefix:
        model_prefix = FLAGS.model_name_prefix + '_'
    return f'{model_prefix}{FLAGS.graph_encoder_model.upper()}_{FLAGS.dataset}_lr{FLAGS.lr}_{FLAGS.link_pred_model}'


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


def perform_inductive_training(model_name, training_data, val_data, inference_data, data, test_edge_bundle,
                               negative_samples, output_dir, representation_size, device, input_size, has_features,
                               g_zoo):
    (test_old_old_ei, test_old_new_ei, test_new_new_ei, test_edge_index) = test_edge_bundle

    model = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes,
                            n_feats=data.x.size(1)).to(device)
    predictor = MLPProdDecoder(representation_size, hidden_size=FLAGS.predictor_hidden_size).to(device)

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


def perform_transductive_training(model_name, data, edge_split, output_dir, representation_size, device, input_size,
                                  has_features: bool, g_zoo):
    model = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes,
                            n_feats=data.x.size(1)).to(device)
    predictor = MLPProdDecoder(representation_size, hidden_size=FLAGS.predictor_hidden_size).to(device)

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
    torch.save((model, predictor), path.join(output_dir, 'model.pt'))
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


######
# Main
######
def main(_):
    if FLAGS.logdir is None:
        new_logdir = f'./runs/{FLAGS.dataset}'
        log.info(f'No logdir set, using default of {new_logdir}')
        FLAGS.logdir = new_logdir

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    g_zoo = EncoderZoo(FLAGS)

    g_zoo.check_model(FLAGS.graph_encoder_model)
    log.info(f'Found link pred validation models: {FLAGS.link_pred_model}')
    log.info(f'Using encoder model: {FLAGS.graph_encoder_model}')

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    wandb.init(project=f'sup-gnn-prod', config={'model_name': get_full_model_name(), **FLAGS.flag_values_dict()})

    # create log directory
    OUTPUT_DIR = os.path.join(FLAGS.logdir, f'{get_full_model_name()}_{wandb.run.id}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load data
    st_time = time.time_ns()
    dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
    data = dataset[0]

    if FLAGS.split_method == 'transductive':
        if isinstance(dataset, PygLinkPropPredDataset):
            edge_split = dataset.get_edge_split()
        else:
            edge_split = do_transductive_edge_split(dataset)
            data.edge_index = edge_split['train']['edge'].t()
        data.to(device)
    else:
        training_data, val_data, inference_data, data, test_edge_bundle, negative_samples = do_node_inductive_edge_split(
            dataset, split_seed=FLAGS.split_seed)

    end_time = time.time_ns()

    log.info(f'Took {(end_time - st_time) / 1e9}s to load data')
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))

    # check to make sure we have node features
    if data.x is None:
        raise ValueError(f'Dataset does not contain node features, which are required.')

    input_size = data.x.size(1)
    representation_size = FLAGS.graph_encoder_layer[-1]

    all_results = []
    for run_num in range(FLAGS.num_runs):
        print('=' * 30)
        print('=' * 30)
        print('=' * 10 + f'  Run #{run_num}  ' + '=' * 10)
        print('=' * 30)
        print('=' * 30)

        if FLAGS.split_method == 'transductive':
            _, _, results = perform_transductive_training(model_name=get_full_model_name(),
                                                          data=data,
                                                          edge_split=edge_split,
                                                          output_dir=OUTPUT_DIR,
                                                          representation_size=representation_size,
                                                          device=device,
                                                          input_size=input_size,
                                                          has_features=True,
                                                          g_zoo=g_zoo)
        else:
            _, _, results = perform_inductive_training(model_name=get_full_model_name(),
                                                       training_data=training_data,
                                                       val_data=val_data,
                                                       inference_data=inference_data,
                                                       data=data,
                                                       test_edge_bundle=test_edge_bundle,
                                                       negative_samples=negative_samples,
                                                       output_dir=OUTPUT_DIR,
                                                       representation_size=representation_size,
                                                       device=device,
                                                       input_size=input_size,
                                                       has_features=True,
                                                       g_zoo=g_zoo)
        all_results.append(results)

    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)

    log.info(f'Done! Run information can be found at {OUTPUT_DIR}')


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
