# This file contains flag-related utility functions
import os
from absl import flags
import logging
from os import path

from .transforms import VALID_TRANSFORMS

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)


def between_zero_one(x):
    """Checks if x is between zero and one. Used by validators."""
    return 0 < x < 1


def define_flags(model_name):
    define_shared_flags(model_name)
    if model_name == 'GRACE' or model_name == 'NCL':
        define_aug_flags()


def define_shared_flags(model_name):
    """Define flags that are shared across all models"""
    get_default = lambda name: get_flag_default(model_name, name)

    # General flags
    flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
    flags.DEFINE_string('model_name_prefix', '', 'Prefix to prepend to the output directory')
    flags.DEFINE_bool(
        'debug', False,
        'Whether or not we are debugging. No effect, just used for logging purposes.')
    flags.DEFINE_integer(
        'num_runs', 5,
        'Number of times to train/evaluate the model and re-run to obtain reliable results.')
    flags.DEFINE_bool('do_classification_eval', False,
                      'Whether or not to evaluate the model\'s classification performance')

    # Dataset flags
    flags.DEFINE_integer('split_seed', 234, 'Seed to use for dataset splitting.')
    flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
    flags.DEFINE_enum('dataset', 'cora', [
        'amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'cora', 'citeseer'
    ], 'Which graph dataset to use.')
    flags.DEFINE_enum('split_method', 'transductive', ['inductive', 'transductive'],
                      'Which method to use to split the dataset (inductive or transductive).')
    flags.DEFINE_float('big_split_ratio', 0.2, 'Inductive split for the big datasets.')

    # Encoder/training flags
    flags.DEFINE_enum('graph_encoder_model', 'gcn', ['gcn'], 'Which type of graph encoder to use.')
    flags.DEFINE_multi_integer('graph_encoder_layer', [256, 256],
                               'Conv layer sizes. Each value indicates a new layer.')
    flags.DEFINE_float('lr', get_default('lr'), 'The learning rate for model training.')
    flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
    flags.DEFINE_integer('epochs', get_default('epochs'), 'The number of training epochs.')

    # Link predictor flags
    flags.DEFINE_multi_enum(
        'link_pred_model', ['prod_mlp'], ['mlp', 'prod_mlp'],
        'Which link prediction model to use (product-based MLP or concat-based MLP).')
    flags.DEFINE_integer('link_mlp_hidden_size', 128,
                         'Size of the hidden layer in the MLP for evaluation')
    flags.DEFINE_float('link_mlp_lr', 0.01, 'Link decoder learning rate.')
    flags.DEFINE_integer('link_nn_epochs', 10000, 'Number of epochs in the NN for evaluation.')
    flags.DEFINE_bool('batch_links', False, 'Whether or not to perform batching on links.')
    flags.DEFINE_integer('link_batch_size', 64 * 1024, 'Batch size for links.')
    flags.DEFINE_enum('trivial_neg_sampling', 'false', ['true', 'false'],
                      'Whether or not to perform trivial random sampling.')
    flags.DEFINE_bool(
        'adjust_layer_sizes', False,
        'Whether or not to adjust MLP layer sizes for fair comparisons. '
        'This will reduce the size of the hidden layer for the concat-based MLP so '
        'that the total number of weights are the same as the product-based MLP.'
    )


def define_aug_flags():
    """Define flags that are specific to augmentation-based models"""
    flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
    flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
    flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
    flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')
    flags.DEFINE_enum('graph_transforms', 'standard', list(VALID_TRANSFORMS.keys()),
                      'Which graph augmentations to use.')

    flags.register_validator('drop_edge_p_1', between_zero_one, 'must be between 0 and 1')
    flags.register_validator('drop_feat_p_1', between_zero_one, 'must be between 0 and 1')
    flags.register_validator('drop_edge_p_2', between_zero_one, 'must be between 0 and 1')
    flags.register_validator('drop_feat_p_2', between_zero_one, 'must be between 0 and 1')


OVERRIDES = {'default': {'epochs': 10000, 'lr': 1e-5}, 'E2E-GCN': {'epochs': 1000, 'lr': 1e-3}}


def get_flag_default(model_name, field_name):
    default_val = OVERRIDES['default'][field_name]
    model_dict = OVERRIDES.get(model_name, None)
    if model_dict is None:
        return default_val

    return model_dict.get(field_name, default_val)


def get_dynamic_defaults():
    if FLAGS.logdir is None:
        new_logdir = f'./runs/{FLAGS.dataset}'
        log.info(f'No logdir set, using default of {new_logdir}')
        FLAGS.logdir = new_logdir


def init_dir_save_flags(model_name):
    output_dir = os.path.join(FLAGS.logdir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(path.join(output_dir, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file
    return output_dir
