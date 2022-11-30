import json
import logging
import time

from absl import app
from absl import flags
import torch
from torch import nn
import wandb
import torch.nn.functional as F

from lib.data import get_dataset
from lib.models import LinkPredictorZoo, EncoderZoo
from lib.training import perform_transductive_margin_training, perform_inductive_margin_training
from lib.eval import do_all_eval, do_inductive_eval
from ogb.linkproppred import PygLinkPropPredDataset

import lib.flags as FlagHelper
from lib.utils import do_node_inductive_edge_split, do_transductive_edge_split, is_small_dset, merge_multirun_results

######
# Flags
######
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FLAGS = flags.FLAGS

# Define shared flags
FlagHelper.define_flags('ML-GCN')

# Flags specific to ML-GCN
flags.DEFINE_float('margin', 3.0, 'Margin used for max-margin loss')
flags.DEFINE_integer('lr_warmup_epochs', 500, 'Warmup period for learning rate.')

flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')

flags.DEFINE_integer('pos_samples', 3, 'Number of positive samples to use for margin loss')
flags.DEFINE_integer('neg_samples', 3, 'Number of negative samples to use for margin loss')
flags.DEFINE_enum('pos_neg_agg_method', 'min_max', ['min_max', 'mean'],
                  'Method used to aggregate margins from pos/neg samples')
flags.DEFINE_bool('normalize_embeddings', False, 'Whether or not to normalize embeddings')


def get_full_model_name():
    model_prefix = ''
    if FLAGS.model_name_prefix:
        model_prefix = FLAGS.model_name_prefix + '_'

    return f'{model_prefix}{FLAGS.graph_encoder_model.upper()}_{FLAGS.dataset}_lr{FLAGS.lr}_m{FLAGS.margin}_{FLAGS.link_pred_model}'


######
# Main
######
def main(_):
    FlagHelper.get_dynamic_defaults()

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

    output_dir = FlagHelper.init_dir_save_flags(get_full_model_name())

    wandb.init(project=f'ml-gcn',
               config={
                   'model_name': get_full_model_name(),
                   **FLAGS.flag_values_dict()
               })

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

    has_features = True
    input_size = data.x.size(1)

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
                data, edge_split, output_dir, device, input_size, has_features, g_zoo)

            if FLAGS.normalize_embeddings:
                log.info('Normalizing embeddings...')
                representations = F.normalize(representations, dim=1)
            embeddings = nn.Embedding.from_pretrained(representations, freeze=True)

            results, _ = do_all_eval(model_name=get_full_model_name(),
                                     output_dir=output_dir,
                                     valid_models=valid_models,
                                     dataset=dataset,
                                     edge_split=edge_split,
                                     embeddings=embeddings,
                                     lp_zoo=lp_zoo,
                                     wb=wandb)
        else:  # inductive
            encoder, representations, time_bundle = perform_inductive_margin_training(
                training_data, val_data, data, output_dir, device, input_size, has_features, g_zoo)

            results = do_inductive_eval(model_name=get_full_model_name(),
                                         output_dir=output_dir,
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

        (total_time, _, _, times) = time_bundle
        all_times.append(times.tolist())
        total_times.append(int(total_time))

        # incremental updates
        with open(f'{output_dir}/times.json', 'w') as f:
            json.dump({'all_times': all_times, 'total_times': total_times}, f)
        all_results.append(results)

    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    with open(f'{output_dir}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)
    log.info(f'Done! Run information can be found at {output_dir}')


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
