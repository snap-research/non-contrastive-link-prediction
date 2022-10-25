import logging
import os
from os import path
import json
import time

from absl import app
from absl import flags
import torch
from torch import nn
from torch_geometric.utils import negative_sampling
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import wandb

from lib.data import get_dataset, get_wiki_cs, ConvertToFloat
from lib.eval import eval_all
from lib.link_predictors import LinkPredictorZoo, MLPProdLinkPredictor
from lib.models import EncoderZoo
from lib.transforms import VALID_TRANSFORMS
from ogb.linkproppred import PygLinkPropPredDataset

from lib.utils import add_node_feats, merge_multirun_results, set_random_seeds, do_edge_split

######
# Flags
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs', [
    'amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'ogbl-collab', 'ogbl-ddi',
    'ogbl-ppa', 'cora', 'citeseer'
], 'Which graph dataset to use.')
flags.DEFINE_enum('link_pred_model', 'mlp', ['mlp', 'prod_mlp', 'dot'], 'Which link prediction model to use')
flags.DEFINE_enum('graph_encoder_model', 'gcn', ['gcn', 'sage'], 'Which graph encoder model to use')
flags.DEFINE_enum('graph_transforms', 'standard', list(VALID_TRANSFORMS.keys()), 'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
flags.DEFINE_bool('eval_only', False, 'Only evaluate the model.')
flags.DEFINE_multi_enum(
    'eval_only_pred_model', [], ['lr', 'mlp', 'cosine', 'seal', 'prod_lr'],
    'Which link prediction models to use (overwrites link_pred_model if eval_only is True and this is set)')

flags.DEFINE_bool('batch_links', False, 'Whether or not to perform batching on links')
flags.DEFINE_integer('link_batch_size', 64 * 1024, 'Batch size for links')
# TODO(author): implement
flags.DEFINE_bool('batch_graphs', False, 'Whether or not to perform batching on graphs')

flags.DEFINE_enum('feature_fallback', 'degree', ['degree', 'learn'],
                  'Which method to use as a fallback if the matrix has no node features')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', [256, 256], 'Conv layer sizes.')
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

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')

# Link prediction model-specific flags
# MLP:
flags.DEFINE_integer('link_mlp_hidden_size', 128, 'Size of hidden layer in MLP for evaluation')
flags.DEFINE_float('link_mlp_lr', 0.01, 'Size of hidden layer in MLP for evaluation')
flags.DEFINE_integer('link_nn_epochs', 10000, 'Number of epochs in the NN for evaluation')
flags.DEFINE_enum('trivial_neg_sampling', 'auto', ['true', 'false', 'auto'],
                  'Whether or not to do trivial random sampling. Auto will choose based on dataset size.')


def get_full_model_name():
    model_prefix = ''
    if FLAGS.model_name_prefix:
        model_prefix = FLAGS.model_name_prefix + '_'
    return f'{model_prefix}{FLAGS.graph_encoder_model.upper()}_{FLAGS.dataset}_lr{FLAGS.lr}_{FLAGS.link_pred_model}'


def perform_training(model_name, dataset, data, edge_split, output_dir, writer, representation_size, device,
                     num_eval_splits, input_size: int, has_features: bool, g_zoo):
    model = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, has_features, data.num_nodes,
                            n_feats=data.x.size(1)).to(device)
    predictor = MLPProdLinkPredictor(representation_size, hidden_size=FLAGS.predictor_hidden_size).to(device)

    # # optimizer
    optimizer = Adam(list(model.parameters()) + list(predictor.parameters()),
                     lr=FLAGS.lr,
                     weight_decay=FLAGS.weight_decay)
    criterion = BCEWithLogitsLoss()

    # we already filtered out test/val edges
    train_edge = data.edge_index

    #####
    # Train & eval functions
    #####
    def full_train(step):
        model.train()
        predictor.train()
        optimizer.zero_grad()

        if not has_features:
            data.x = model.get_node_feats().weight.data.clone().detach()

        model.train()
        optimizer.zero_grad()

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(edge_index=data.edge_index,
                                           num_nodes=dataset[0].num_nodes,
                                           num_neg_samples=train_edge.size(1),
                                           method='sparse')

        edge_label_index = torch.cat(
            [train_edge, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([train_edge.new_ones(train_edge.size(1)),
                                train_edge.new_zeros(neg_edge_index.size(1))],
                               dim=0)

        model_out = model(data)
        edge_embeddings = model_out[edge_label_index]
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
        out = predictor(combined)

        loss = criterion(out.view(-1), edge_label.float())
        loss.backward()
        optimizer.step()
        return loss

    def batched_train(step):
        model.train()
        predictor.train()

        pos_train_edge = edge_split['train']['edge'].to(device)
        total_loss = total_examples = 0

        for perm in DataLoader(range(pos_train_edge.size(0)), FLAGS.link_batch_size, shuffle=True):
            optimizer.zero_grad()
            model_out = model(data)

            pos_edge = pos_train_edge[perm].T
            pos_combined = torch.hstack((model_out[pos_edge[0]], model_out[pos_edge[1]]))
            pos_out = predictor(pos_combined).view(-1)

            if FLAGS.trivial_neg_sampling == 'true':
                neg_edge = torch.randint(0, data.num_nodes, pos_edge.size(), dtype=torch.long, device=device)
            elif FLAGS.trivial_neg_sampling == 'false':
                neg_edge = negative_sampling(data.edge_index, data.num_nodes, pos_edge.size(1))
            else:
                raise ValueError('Invalid flag value for trivial_neg_sampling...')

            neg_combined = torch.hstack((model_out[neg_edge[0]], model_out[pos_edge[1]]))
            neg_out = predictor(neg_combined).view(-1)
            all_out = torch.cat([pos_out, neg_out])

            edge_label = torch.cat([train_edge.new_ones(pos_edge.size(1)),
                                    train_edge.new_zeros(pos_edge.size(1))],
                                   dim=0)

            loss = criterion(all_out.view(-1), edge_label.float())
            loss.backward()

            optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def batched_eval(eval_edge, eval_edge_neg):
        model.eval()
        predictor.eval()
        model_out = model(data)

        pos_preds = []
        for perm in DataLoader(range(eval_edge.size(1)), FLAGS.link_batch_size):
            pos_edge = eval_edge[:, perm]
            pos_combined = torch.hstack((model_out[pos_edge[0]], model_out[pos_edge[1]]))
            pos_out = predictor(pos_combined).view(-1)
            pos_preds.append(pos_out.squeeze().cpu())

        neg_preds = []
        for perm in DataLoader(range(eval_edge_neg.size(1)), FLAGS.link_batch_size):
            neg_edge = eval_edge_neg[:, perm]
            neg_combined = torch.hstack((model_out[neg_edge[0]], model_out[neg_edge[1]]))
            neg_out = predictor(neg_combined).view(-1)
            neg_preds.append(neg_out.squeeze().cpu())

        pos_pred = torch.cat(pos_preds, dim=0)
        neg_pred = torch.cat(neg_preds, dim=0)
        return eval_all(pos_pred, neg_pred)

    @torch.no_grad()
    def eval_model(eval_edge, eval_edge_neg):
        model.eval()
        predictor.eval()
        n_edges = eval_edge.size(1)

        edge_label_index = torch.cat(
            [eval_edge, eval_edge_neg],
            dim=-1,
        )

        model_out = model(data)
        embeddings = nn.Embedding.from_pretrained(model_out, freeze=True)
        edge_embeddings = embeddings(edge_label_index)
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
        out = predictor.predict(combined).view(-1)

        y_pred_pos, y_pred_neg = out[:n_edges], out[n_edges:]
        return eval_all(y_pred_pos, y_pred_neg)

    #####
    ## End train & eval functions
    #####

    valid_edge, test_edge = edge_split['valid']['edge'].T.to(device), edge_split['test']['edge'].T.to(device)
    valid_edge_neg, test_edge_neg = edge_split['valid']['edge_neg'].T.to(device), edge_split['test']['edge_neg'].T.to(
        device)

    best_val = None
    best_results = None
    target_metric = 'hits@50'
    last_epoch = None

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        if FLAGS.batch_links:
            train_loss = full_train(epoch - 1)
            val_res = eval_model(valid_edge, valid_edge_neg)
            test_res = eval_model(test_edge, test_edge_neg)
            # print('Validation:', val_res)
            # print('Test:', test_res)
            metric_names = None
        else:
            train_loss = batched_train(epoch - 1)
            val_res = batched_eval(valid_edge, valid_edge_neg)
            test_res = batched_eval(test_edge, test_edge_neg)
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

    results_path = path.join(output_dir, 'output.json')
    if path.exists(results_path):
        log.info('Existing file found, appending results')
        with open(results_path, 'rb') as f:
            contents = json.load(f)
        log.debug(f'Existing contents: {contents}')

        contents['results'].extend(all_results)

        mn = model_name
        if contents['model_name'] != mn:
            log.warn(f'[WARNING]: Model names do not match - {contents["model_name"]} vs {mn}')

        with open(results_path, 'w') as f:
            json.dump(
                {
                    'model_name': mn,
                    'results': contents['results'],
                    # 'class_results': classification_results
                },
                f,
                indent=4)
        log.info(f'Appended results to {results_path}')
    else:
        log.info('No results file found, writing to new one')
        with open(results_path, 'w') as f:
            json.dump(
                {
                    'model_name': model_name,
                    'results': all_results,
                    # 'class_results': classification_results
                },
                f,
                indent=4)
        log.info(f'Wrote results to file at {results_path}')

    return model, predictor, all_results

    # save encoder weights
    # torch.save({'model': model.online_encoder.state_dict()}, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'))
    # encoder = copy.deepcopy(model.online_encoder.eval())
    # representations = compute_data_representations_only(encoder, data, device, has_features=has_features)
    # torch.save(representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt'))

    # return encoder, representations


######
# Main
######
def main(_):
    if FLAGS.eval_only_pred_model and FLAGS.eval_only:
        log.info(
            f'Overridding current value of eval_only_pred_model ({FLAGS.link_pred_model}) with {FLAGS.eval_only_pred_model}'
        )
        FLAGS.link_pred_model = FLAGS.eval_only_pred_model

    if FLAGS.logdir is None:
        new_logdir = f'./runs/{FLAGS.dataset}'
        log.info(f'No logdir set, using default of {new_logdir}')
        FLAGS.logdir = new_logdir

    if FLAGS.trivial_neg_sampling == 'auto':
        if FLAGS.dataset == 'ogbl-collab':
            FLAGS.trivial_neg_sampling = 'true'
            log.info(f'Setting trivial_neg_sampling to true since auto is set and the dataset is large')
        else:
            FLAGS.trivial_neg_sampling = 'false'
            log.info(f'Setting trivial_neg_sampling to true since auto is set and the dataset is small')
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))
    lp_zoo = LinkPredictorZoo(FLAGS)
    g_zoo: EncoderZoo = EncoderZoo(FLAGS)

    g_zoo.check_model(FLAGS.graph_encoder_model)
    valid_models = lp_zoo.filter_models(FLAGS.link_pred_model)
    log.info(f'Found link pred validation models: {FLAGS.link_pred_model}')
    log.info(f'Using encoder model: {FLAGS.graph_encoder_model}')

    if len(valid_models) > 1:
        raise NotImplementedError('Currently, only one type of NN link pred model can be used at once')

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    wandb.init(project=f'sup-gnn', config={'model_name': get_full_model_name(), **FLAGS.flag_values_dict()})

    # create log directory
    OUTPUT_DIR = os.path.join(FLAGS.logdir, f'{get_full_model_name()}_{wandb.run.id}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path.join(OUTPUT_DIR, 'eval_config.cfg' if FLAGS.eval_only else 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # load data
    st_time = time.time_ns()
    if FLAGS.dataset != 'wiki-cs':
        # TODO(author): figure out a better way to do this
        if FLAGS.dataset == 'obgl-ddi':
            # no features to normalize
            dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset, transform=ConvertToFloat())
        else:
            dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, _, _ = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    data = dataset[0]  # all dataset include one graph

    if isinstance(dataset, PygLinkPropPredDataset):
        # TODO(author): move it lower once we're sure this works properly
        edge_split = dataset.get_edge_split()
    else:
        edge_split = do_edge_split(dataset)
        data.edge_index = edge_split['train']['edge'].t()

    end_time = time.time_ns()
    log.info(f'Took {(end_time - st_time) / 1e9}s to load data')

    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))

    # only move data if we're doing full batch
    if not FLAGS.batch_links:
        data = data.to(device)

    # setup tensorboard and make custom layout
    writer = SummaryWriter(OUTPUT_DIR)
    layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
    writer.add_custom_scalars(layout)

    # build networks
    if data.x is None:
        if FLAGS.feature_fallback == 'degree':
            has_features = True
            log.warn(
                f'[WARNING] Dataset {FLAGS.dataset} appears to be featureless - using one-hot degree matrix as features'
            )

            # data.x = torch.ones((input_size, input_size)).to(device)
            data = add_node_feats(data, device)
            input_size = data.x.size(1)
        elif FLAGS.feature_fallback == 'learn':
            has_features = False
            input_size = FLAGS.graph_encoder_layer[0]
            log.warn(f'[WARNING] Dataset {FLAGS.dataset} appears to be featureless - using learnable feature matrix')
        else:
            raise ValueError(f'Unknown value for feature_fallback: {FLAGS.feature_fallback}')
    else:
        has_features = True
        input_size = data.x.size(1)
    representation_size = FLAGS.graph_encoder_layer[-1]

    all_results = []

    for run_num in range(FLAGS.num_runs):
        print('=' * 30)
        print('=' * 30)
        print('=' * 10 + f'  Run #{run_num}  ' + '=' * 10)
        print('=' * 30)
        print('=' * 30)

        model, predictor, results = perform_training(get_full_model_name(), dataset, data, edge_split, OUTPUT_DIR,
                                                     writer, representation_size, device, num_eval_splits, input_size,
                                                     has_features, g_zoo)
        all_results.append(results)

    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)

    log.info(f'Done! Run information can be found at {OUTPUT_DIR}')


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
