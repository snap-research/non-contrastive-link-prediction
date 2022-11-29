import json
import logging
import os
from os import path
import time

from absl import app
from absl import flags
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.nn.functional as F

from lib.data import get_dataset
from lib.link_predictors import LinkPredictorZoo
from lib.models import EncoderZoo
from lib.training import perform_transductive_margin_training, perform_inductive_margin_training
from lib.eval import do_production_eval
from ogb.linkproppred import PygLinkPropPredDataset
from lib.transforms import VALID_TRANSFORMS

from lib.utils import add_node_feats, do_node_inductive_edge_split, do_transductive_edge_split, is_small_dset, merge_multirun_results, set_random_seeds

######
# Flags
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3,
                     'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_string('model_name_prefix', '', 'Prefix to prepend in front of the model name.')
flags.DEFINE_enum('dataset', 'coauthor-cs', [
    'amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs',
    'ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'cora', 'citeseer'
], 'Which graph dataset to use.')
flags.DEFINE_multi_enum('link_pred_model', ['prod_mlp'],
                        ['lr', 'mlp', 'cosine', 'seal', 'prod_lr', 'prod_mlp'],
                        'Which link prediction model to use')
flags.DEFINE_enum('graph_encoder_model', 'gcn', ['gcn', 'sage', 'gat', 'gin'],
                  'Which graph encoder model to use')
flags.DEFINE_enum('graph_transforms', 'standard', list(VALID_TRANSFORMS.keys()),
                  'Which graph dataset to use.')

flags.DEFINE_integer('num_runs', 5, 'Number of times to train/evaluate the model and re-run')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
flags.DEFINE_bool('eval_only', False, 'Only evaluate the model.')
flags.DEFINE_multi_enum(
    'eval_only_pred_model', [], ['lr', 'mlp', 'cosine', 'prod_mlp', 'prod_lr'],
    'Which link prediction models to use (overwrites link_pred_model if eval_only is True and this is set)'
)
flags.DEFINE_float('margin', 3.0, 'Margin used for max-margin loss')
flags.DEFINE_integer('split_seed', 234, 'Seed to use for dataset splitting')

flags.DEFINE_enum('feature_fallback', 'degree', ['degree', 'learn'],
                  'Which method to use as a fallback if the matrix has no node features')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', [256, 128], 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 500, 'Warmup period for learning rate.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')

# Link prediction model-specific flags
# MLP:
flags.DEFINE_integer('link_mlp_hidden_size', 128, 'Size of hidden layer in MLP for evaluation')
flags.DEFINE_float('link_mlp_lr', 0.01, 'Learning rate for the MLP during evaluation')
flags.DEFINE_float('big_split_ratio', 0.2, 'Ratio for big dataset inductive split')
flags.DEFINE_integer('link_nn_epochs', 10000, 'Number of epochs in the NN for evaluation')

flags.DEFINE_integer('neg_samples', 3, 'Number of negative samples to use for margin loss')
flags.DEFINE_integer('pos_samples', 3, 'Number of positive samples to use for margin loss')
flags.DEFINE_enum('pos_neg_agg_method', 'min_max', ['min_max', 'mean'],
                  'Method used to aggregate margins from pos/neg samples')
flags.DEFINE_bool('debug', False, 'Whether or not this is a debugging run')
flags.DEFINE_bool('normalize_embeddings', False, 'Whether or not to normalize embeddings')
flags.DEFINE_bool('do_classification_eval', False, 'Whether or not to do classification eval')
flags.DEFINE_bool('batch_links', False,
                  'Whether or not to batch links (not implemented for training yet)')
flags.DEFINE_enum('split_method', 'transductive', ['inductive', 'transductive'],
                  'Which method to use to split the dataset (inductive or transductive).')


def get_full_model_name():
    model_prefix = ''
    if FLAGS.model_name_prefix:
        model_prefix = FLAGS.model_name_prefix + '_'

    return f'{model_prefix}{FLAGS.graph_encoder_model.upper()}_{FLAGS.dataset}_lr{FLAGS.lr}_m{FLAGS.margin}_{FLAGS.link_pred_model}'


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

    wandb.init(project=f'ml-gcn-{FLAGS.split_method}',
               config={
                   'model_name': get_full_model_name(),
                   **FLAGS.flag_values_dict()
               })

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
        raise NotImplementedError(
            'Currently, only one type of NN link pred model can be used at once')

    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    # create log directory
    OUTPUT_DIR = os.path.join(FLAGS.logdir, get_full_model_name())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path.join(OUTPUT_DIR, 'eval_config.cfg' if FLAGS.eval_only else 'config.cfg'),
              "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # load data
    st_time = time.time_ns()
    dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)

    data = dataset[0]  # all dataset include one graph

    if FLAGS.split_method == 'transductive':
        edge_split = do_transductive_edge_split(dataset)
        data.edge_index = edge_split['train']['edge'].t()
        data = data.to(device)
    else:  # inductive
        if isinstance(dataset, PygLinkPropPredDataset):
            raise NotImplementedError()
        else:
            training_data, val_data, inference_data, data, test_edge_bundle, negative_samples = do_node_inductive_edge_split(
                dataset=dataset,
                split_seed=FLAGS.split_seed,
                small_dataset=is_small_dset(FLAGS.dataset))  # type: ignore

    end_time = time.time_ns()
    log.info(f'Took {(end_time - st_time) / 1e9}s to load data')

    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))

    # build networks
    if data.x is None:
        if FLAGS.feature_fallback == 'degree':
            has_features = True
            log.warn(
                f'[WARNING] Dataset {FLAGS.dataset} appears to be featureless - using one-hot degree matrix as features'
            )

            data = add_node_feats(data, device)
            input_size = data.x.size(1)
        elif FLAGS.feature_fallback == 'learn':
            has_features = False
            input_size = FLAGS.graph_encoder_layer[0]
            log.warn(
                f'[WARNING] Dataset {FLAGS.dataset} appears to be featureless - using learnable feature matrix'
            )
        else:
            raise ValueError(f'Unknown value for feature_fallback: {FLAGS.feature_fallback}')
    else:
        has_features = True
        input_size = data.x.size(1)
    FLAGS.graph_encoder_layer[-1]

    all_results = []
    all_times = []
    total_times = []
    time_bundle = None

    for run_num in range(FLAGS.num_runs):
        print('=' * 30)
        print('=' * 30)
        print('=' * 10 + f'  Run #{run_num}  ' + '=' * 10)
        print('=' * 30)
        print('=' * 30)

        if FLAGS.split_method == 'transductive':
            encoder, representations, time_bundle = perform_transductive_margin_training(
                data, edge_split, OUTPUT_DIR, device, input_size, has_features, g_zoo)

            if FLAGS.normalize_embeddings:
                log.info('Normalizing embeddings...')
                representations = F.normalize(representations, dim=1)
            embeddings = nn.Embedding.from_pretrained(representations, freeze=True)
        else:  # inductive
            encoder, representations, time_bundle = perform_inductive_margin_training(
                training_data, val_data, data,
                OUTPUT_DIR, device, input_size, has_features, g_zoo)

            results = do_production_eval(model_name=get_full_model_name(),
                                        output_dir=OUTPUT_DIR,
                                        encoder=encoder,
                                        valid_models=valid_models,
                                        train_data=training_data,
                                        val_data=val_data,
                                        inference_data=inference_data,
                                        lp_zoo=lp_zoo,
                                        device=device,
                                        test_edge_bundle=test_edge_bundle,
                                        negative_samples=negative_samples,
                                        wb=wandb)
        log.info('Finished training!')

        (total_time, std_time, mean_time, times) = time_bundle
        all_times.append(times.tolist())
        total_times.append(int(total_time))

        # incremental updates
        with open(f'{OUTPUT_DIR}/times.json', 'w') as f:
            json.dump({'all_times': all_times, 'total_times': total_times}, f)

        all_results.append(results)

    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)

    log.info(f'Done! Run information can be found at {OUTPUT_DIR}')


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
