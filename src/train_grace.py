import json
import os
import time
import wandb

from absl import app
from absl import flags
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from lib.data import get_dataset
from ogb.linkproppred import PygLinkPropPredDataset

from grace.model import Encoder, Model, drop_feature
from lib.eval import do_all_eval
from lib.link_predictors import LinkPredictorZoo
from lib.training import get_time_bundle

from lib.utils import do_transductive_edge_split, merge_multirun_results  # type: ignore

######
# Flags
######
FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', 'citeseer', [
    'amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'ogbl-collab', 'ogbl-ddi',
    'ogbl-ppa', 'cora', 'citeseer', 'squirrel', 'chameleon', 'crocodile', 'texas'
], 'Which graph dataset to use.')
flags.DEFINE_enum('activation_type', 'prelu', ['prelu', 'relu'], 'Which activation type to use')
flags.DEFINE_enum('graph_encoder_model', 'GCNConv', ['GCNConv'], 'Which type of graph encoder to use')

flags.DEFINE_string('model_name_prefix', '', 'Prefix to prepend to the output directory')
flags.DEFINE_integer('split_seed', 234, 'Seed to use for dataset splitting')
flags.DEFINE_multi_enum('link_pred_model', ['prod_mlp'], ['lr', 'mlp', 'cosine', 'seal', 'prod_lr', 'prod_mlp'],
                        'Which link prediction model to use')
flags.DEFINE_bool('do_classification_eval', True, 'Whether or not to evaluate the model\'s classification performance')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_bool('batch_links', False, 'Whether or not to perform batching on links')
flags.DEFINE_bool('debug', False, 'Whether or not we are debugging. No effect, just used for logging purposes')

flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_integer('link_mlp_hidden_size', 128, 'Size of hidden layer in MLP for evaluation')
flags.DEFINE_float('link_mlp_lr', 0.01, 'Size of hidden layer in MLP for evaluation')
flags.DEFINE_integer('link_nn_epochs', 8000, 'Number of epochs in the NN for evaluation')
flags.DEFINE_integer('num_runs', 5, 'Number of times to train/evaluate the model and re-run')

flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')
flags.DEFINE_float('tau', 0., 'GRACE parameter')


def train(model: Model, optimizer, x, edge_index, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1,
          drop_feature_rate_2):
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


def main(_):
    model_prefix = ''
    if FLAGS.model_name_prefix:
        model_prefix = f'{FLAGS.model_name_prefix}_'
    model_name = f'{model_prefix}GRACE_{FLAGS.dataset}'
    assert (FLAGS.drop_edge_p_1 != 0 and FLAGS.drop_edge_p_2 != 0 and FLAGS.drop_feat_p_1 != 0 and
            FLAGS.drop_feat_p_2 != 0)

    wandb.init(project='grace', config={'model_name': model_name, **FLAGS.flag_values_dict()})
    if wandb.run is None:
        raise ValueError('Failed to initialize wandb run!')

    OUTPUT_DIR = os.path.join('./runs', FLAGS.dataset, f'{model_name}_{wandb.run.id}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    learning_rate = FLAGS.lr
    num_hidden = 256
    num_proj_hidden = 256
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[FLAGS.activation_type]
    base_model = ({'GCNConv': GCNConv})[FLAGS.graph_encoder_model]
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

    if isinstance(dataset, PygLinkPropPredDataset):
        # TODO(author): move it lower once we're sure this works properly
        edge_split = dataset.get_edge_split()
    else:
        edge_split = do_transductive_edge_split(dataset, FLAGS.split_seed)
        data.edge_index = edge_split['train']['edge'].t()  # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)  # type: ignore

    # TODO(author): change to use EncoderZoo?
    lp_zoo = LinkPredictorZoo(FLAGS)
    valid_models = lp_zoo.filter_models(FLAGS.link_pred_model)

    all_results = []
    all_class_results = []
    all_times = []
    total_times = []

    for run_num in range(FLAGS.num_runs):
        print('=' * 30)
        print('=' * 30)
        print('=' * 10 + f'  Run #{run_num}  ' + '=' * 10)
        print('=' * 30)
        print('=' * 30)

        encoder = Encoder(dataset.num_features, num_hidden, activation, base_model=base_model, k=num_layers).to(device)
        model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        times = []

        for epoch in range(1, num_epochs + 1):
            st_time = time.time_ns()
            loss = train(model=model,
                         optimizer=optimizer,
                         x=data.x,
                         edge_index=data.edge_index,
                         drop_edge_rate_1=drop_edge_rate_1,
                         drop_edge_rate_2=drop_edge_rate_2,
                         drop_feature_rate_1=drop_feature_rate_1,
                         drop_feature_rate_2=drop_feature_rate_2)
            elapsed = time.time_ns() - st_time
            times.append(elapsed)

            if epoch % 25 == 0:
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        time_bundle = get_time_bundle(times)
        (total_time, std_time, mean_time, times) = time_bundle
        all_times.append(times.tolist())
        total_times.append(int(total_time))

        # incremental updates
        with open(f'{OUTPUT_DIR}/times.json', 'w') as f:
            json.dump({'all_times': all_times, 'total_times': total_times}, f)

        representations = model(data.x, data.edge_index)
        embeddings = nn.Embedding.from_pretrained(representations, freeze=True)

        print("=== Final ===")
        results, classification_results = do_all_eval(model_name,
                                                      output_dir=OUTPUT_DIR,
                                                      valid_models=valid_models,
                                                      dataset=dataset,
                                                      edge_split=edge_split,
                                                      embeddings=embeddings,
                                                      lp_zoo=lp_zoo,
                                                      wb=wandb)

        all_results.append(results)
        all_class_results.append(classification_results)

    agg_results, to_log = merge_multirun_results(all_results)
    wandb.log(to_log)

    with open(f'{OUTPUT_DIR}/agg_results.json', 'w') as f:
        json.dump(agg_results, f)

    print(f'Done! Run information can be found at {OUTPUT_DIR}')


if __name__ == "__main__":
    print('PyTorch version: %s' % torch.__version__)
    app.run(main)
