import logging
import os
from os import path
import time
import json
from absl import app
from absl import flags
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from lib.data import get_dataset, get_wiki_cs, ConvertToFloat
from lib.bgrl import compute_data_representations_only
from lib.link_predictors import LinkPredictorZoo
from lib.models import EncoderZoo
from lib.eval import do_all_eval, do_production_eval, perform_nn_link_eval
from ogb.linkproppred import PygLinkPropPredDataset
from lib.training import perform_bgrl_training, perform_cca_ssg_training, perform_gbt_training, perform_triplet_training
import wandb
from lib.transforms import VALID_NEG_TRANSFORMS, VALID_TRANSFORMS
from lib.utils import add_node_feats, do_node_inductive_edge_split, do_transductive_edge_split, is_small_dset, merge_multirun_results, set_random_seeds
import lib.flags as FlagHelper

######
# Flags
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

FLAGS = flags.FLAGS

FlagHelper.define_flags('NCL')
# Dataset.
flags.DEFINE_enum('base_model', 'bgrl', ['gbt', 'bgrl', 'triplet', 'cca'], 'Which base model to use.')
flags.DEFINE_enum('scheduler', 'cosine', ['cyclic', 'cosine'], 'Which lr scheduler to use')

flags.DEFINE_enum('negative_transforms', 'randomize-feats', list(VALID_NEG_TRANSFORMS.keys()),
                  'Which negative graph transforms to use (triplet formulation only).')
flags.DEFINE_bool('eval_only', False, 'Only evaluate the model.')
flags.DEFINE_multi_enum(
    'eval_only_pred_model', [], ['lr', 'mlp', 'cosine', 'seal', 'prod_lr'],
    'Which link prediction models to use (overwrites link_pred_model if eval_only is True and this is set)')

flags.DEFINE_bool('batch_graphs', False, 'Whether or not to perform batching on graphs')
flags.DEFINE_integer('graph_batch_size', 1024, 'Number of subgraphs to use per minibatch')
flags.DEFINE_integer('graph_eval_batch_size', 128, 'Number of subgraphs to use per minibatch')
flags.DEFINE_integer('n_workers', 0, 'Number of workers to use')

# Architecture.
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_float('cyclic_lr', 0.1, 'The learning rate for model training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')
flags.DEFINE_bool('training_early_stop', False, 'Whether or not to perform early stopping on the training loss')
flags.DEFINE_integer('training_early_stop_patience', 50, 'Training early stopping patience')

# Augmentations.
flags.DEFINE_float('add_edge_ratio_1', 0.,
                   'Ratio of negative edges to sample (compared to existing positive edges) for online net.')
flags.DEFINE_float('add_edge_ratio_2', 0.,
                   'Ratio of negative edges to sample (compared to existing positive edges) for target net.')
flags.DEFINE_float('neg_lambda', 0.5, 'Weight to use for the negative triplet head. Between 0 and 1')

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')

# Link prediction model-specific flags
# MLP:
flags.DEFINE_bool('save_extra', False, 'Whether or not to save extra plotting/debugging info')
flags.DEFINE_bool('dataset_fixed', True, 'Whether or not a message-passing vs normal edges bug was fixed')
flags.DEFINE_bool('adjust_layer_sizes', False, 'Whether or not to adjust MLP layer sizes for fair comparisons')

flags.DEFINE_float('cca_lambda', 0., 'Lambda for CCA-SSG')


def get_full_model_name():
    model_prefix = 'I'
    edge_prob_str = f'dep1{FLAGS.drop_edge_p_1}_dfp1{FLAGS.drop_feat_p_1}_dep2{FLAGS.drop_edge_p_2}_dfp2{FLAGS.drop_feat_p_2}'
    if FLAGS.model_name_prefix:
        model_prefix = FLAGS.model_name_prefix + '_' + model_prefix

    if FLAGS.base_model == 'gbt':
        return f'{model_prefix}GBT_{FLAGS.dataset}_lr{FLAGS.lr}_mm{FLAGS.mm}_{edge_prob_str}'
    elif FLAGS.base_model == 'triplet':
        return f'{model_prefix}TBGRL_{FLAGS.dataset}_lr{FLAGS.lr}_mm{FLAGS.mm}_{edge_prob_str}'

    return f'{model_prefix}BGRL_{FLAGS.dataset}_lr{FLAGS.lr}_mm{FLAGS.mm}_{edge_prob_str}'

######
# Main
######
def main(_):
    log.info('Run started!')

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

    wandb.init(project=f'fixed-{FLAGS.base_model}-prod',
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

    if wandb.run is None:
        raise ValueError('Failed to initialize wandb run!')

    # create log directory
    OUTPUT_DIR = os.path.join(FLAGS.logdir, f'{get_full_model_name()}_{wandb.run.id}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # add config flagfile
    with open(path.join(OUTPUT_DIR, 'eval_config.cfg' if FLAGS.eval_only else 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # save config in JSON
    with open(path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(FLAGS.flag_values_dict(), f)

    # load data
    st_time = time.time_ns()
    dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
    data = dataset[0]  # all datasets (currently) are just 1 graph

    small_dataset = is_small_dset(FLAGS.dataset)
    if small_dataset:
        log.info('Small dataset detected, will use small dataset settings for inductive split.')

    if isinstance(dataset, PygLinkPropPredDataset):
        raise NotImplementedError()

    if FLAGS.split_method == 'transductive':
        edge_split = do_transductive_edge_split(dataset, FLAGS.split_seed)
        data.edge_index = edge_split['train']['edge'].t()  # type: ignore
        data.to(device)
        training_data = data
    else: # inductive
        training_data, val_data, inference_data, data, test_edge_bundle, negative_samples = do_node_inductive_edge_split(
            dataset=dataset, split_seed=FLAGS.split_seed, small_dataset=small_dataset)  # type: ignore

    end_time = time.time_ns()
    log.info(f'Took {(end_time - st_time) / 1e9}s to load data')

    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))

    # only move data if we're doing full batch
    if not FLAGS.batch_graphs:
        training_data = training_data.to(device)

    # build networks
    has_features = True
    input_size = data.x.size(1)  # type: ignore
    representation_size = FLAGS.graph_encoder_layer[-1]

    train_cb = None

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

        if FLAGS.base_model == 'bgrl':
            encoder, representations, time_bundle = perform_bgrl_training(data=training_data,
                                                                          output_dir=OUTPUT_DIR,
                                                                          representation_size=representation_size,
                                                                          device=device,
                                                                          input_size=input_size,
                                                                          has_features=has_features,
                                                                          g_zoo=g_zoo,
                                                                          train_cb=train_cb,
                                                                          extra_return=FLAGS.save_extra)
            if FLAGS.save_extra:
                predictor = representations
            log.info('Finished training!')
        elif FLAGS.base_model == 'cca':
            time_bundle = None
            encoder, representations, time_bundle = perform_cca_ssg_training(data=training_data,
                                                                             output_dir=OUTPUT_DIR,
                                                                             device=device,
                                                                             input_size=input_size,
                                                                             has_features=has_features,
                                                                             g_zoo=g_zoo)
            log.info('Finished training!')
        elif FLAGS.base_model == 'gbt':
            encoder, representations, time_bundle = perform_gbt_training(training_data, OUTPUT_DIR, device,
                                                                         input_size, has_features, g_zoo)
            # del encoder
            log.info('Finished training')
        elif FLAGS.base_model == 'triplet':
            encoder, representations, time_bundle = perform_triplet_training(data=training_data.to(device),
                                                                             output_dir=OUTPUT_DIR,
                                                                             representation_size=representation_size,
                                                                             device=device,
                                                                             input_size=input_size,
                                                                             has_features=has_features,
                                                                             g_zoo=g_zoo,
                                                                             train_cb=train_cb)
        else:
            raise NotImplementedError()

        if time_bundle is not None:
            (total_time, std_time, mean_time, times) = time_bundle
            all_times.append(times.tolist())
            total_times.append(int(total_time))

        if FLAGS.split_method == 'transductive':
            embeddings = nn.Embedding.from_pretrained(representations, freeze=True)
            results, _ = do_all_eval(get_full_model_name(),
                                                output_dir=OUTPUT_DIR,
                                                valid_models=valid_models,
                                                dataset=dataset,
                                                edge_split=edge_split,
                                                embeddings=embeddings,
                                                lp_zoo=lp_zoo,
                                                wb=wandb)
        else: # inductive
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
                                        wb=wandb,
                                        return_extra=FLAGS.save_extra)

        if FLAGS.save_extra:
            nn_model, results = results
        all_results.append(results)

    if FLAGS.save_extra:
        torch.save(
            {
                'nn_model': nn_model.state_dict(),
                'predictor': predictor.state_dict(),
                'encoder': encoder.state_dict()
            }, path.join(OUTPUT_DIR, 'extra_data.pt'))
        torch.save((training_data, val_data, inference_data, data, test_edge_bundle, negative_samples),
                   path.join(OUTPUT_DIR, 'data_split.pt'))
    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    if time_bundle is not None:
        with open(f'{OUTPUT_DIR}/times.json', 'w') as f:
            json.dump({'all_times': all_times, 'total_times': total_times}, f)

    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)

    log.info(f'Done! Run information can be found at {OUTPUT_DIR}')


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
