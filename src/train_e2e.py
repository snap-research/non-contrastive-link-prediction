import logging
import os
import json
import time

from absl import app
from absl import flags
import torch
import wandb

from lib.data import get_dataset
from lib.models import EncoderZoo
from lib.training import perform_e2e_transductive_training, perform_e2e_inductive_training
from ogb.linkproppred import PygLinkPropPredDataset
import lib.flags as FlagHelper

from lib.utils import do_transductive_edge_split, do_node_inductive_edge_split, merge_multirun_results

######
# Flags
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FLAGS = flags.FLAGS

# Define shared flags
FlagHelper.define_flags('E2E-GCN')
# flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')


def get_full_model_name():
    model_prefix = ''
    if FLAGS.model_name_prefix:
        model_prefix = FLAGS.model_name_prefix + '_'
    return f'{model_prefix}{FLAGS.graph_encoder_model.upper()}_{FLAGS.dataset}_lr{FLAGS.lr}_{FLAGS.link_pred_model}'

######
# Main
######
def main(_):
    FlagHelper.get_dynamic_defaults()
    FLAGS.link_pred_model = FLAGS.link_pred_model[0]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    g_zoo = EncoderZoo(FLAGS)

    g_zoo.check_model(FLAGS.graph_encoder_model)
    log.info(f'Found link pred validation models: {FLAGS.link_pred_model}')
    log.info(f'Using encoder model: {FLAGS.graph_encoder_model}')

    wandb.init(project=f'sup-gnn', config={'model_name': get_full_model_name(), **FLAGS.flag_values_dict()})

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
            _, _, results = perform_e2e_transductive_training(model_name=get_full_model_name(),
                                                          data=data,
                                                          edge_split=edge_split,
                                                          output_dir=OUTPUT_DIR,
                                                          representation_size=representation_size,
                                                          device=device,
                                                          input_size=input_size,
                                                          has_features=True,
                                                          g_zoo=g_zoo)
        else:
            _, _, results = perform_e2e_inductive_training(model_name=get_full_model_name(),
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
