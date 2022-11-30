import os
import time
import torch
from absl import flags
from torch.optim import AdamW
from torch_sparse import SparseTensor
from tqdm import tqdm
import wandb
import logging
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from torch_cluster import random_walk

from .utils import get_time_bundle
from ..scheduler import CosineDecayScheduler
from ..utils import compute_data_representations_only


FLAGS = flags.FLAGS
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def edge2lookup(edges, n_nodes, n_samples):
    """Utility function for margin loss negative sampling.
    Takes in the negative edges, the number of nodes, and the number of desired
    samples per node.
    Returns a torch.Tensor containing the negative neighbors for each node, as given
    by the negative edges passed in.
    """
    out = torch.ones((n_nodes, n_samples), dtype=torch.long, device=edges.device) * -1
    for edge in edges.T:
        l, r = edge
        for k in range(n_samples):
            if torch.isnan(out[l, k]):
                out[l, k] = r
                break

    is_nan = out == -1
    to_gen = is_nan.sum()
    # fill the ones we couldn't find negative edges for
    out[is_nan] = torch.randint(
        0, n_nodes, (to_gen,), dtype=torch.long, device=out.device
    )
    return out


def compute_margin_loss(device, n_nodes, edge_index, row, col, model_out):
    """Performs negative sampling and computes the margin loss on the graph represented by
    edge_index.
    It uses the embeddings produced by the model (`model_out`), which should contain autograd
    information to allow for backprop of the loss.
    """

    neg_dists = None
    for _ in range(FLAGS.neg_samples):
        neg_edges = negative_sampling(edge_index, n_nodes, n_nodes)
        lookup = edge2lookup(neg_edges, n_nodes, 1)

        neg_embed = model_out[lookup[:, 0]]
        new_dists = F.logsigmoid(F.cosine_similarity(model_out, neg_embed))
        if neg_dists is None:
            neg_dists = new_dists
        elif FLAGS.pos_neg_agg_method == 'min_max':
            neg_dists = torch.maximum(new_dists, neg_dists)
        else:
            neg_dists += new_dists

    pos_dists = None
    node_vec = torch.arange(0, n_nodes).to(device)
    for _ in range(FLAGS.pos_samples):
        pos_batch = random_walk(row, col, node_vec, walk_length=1, coalesced=False)[
            :, 1
        ]
        pos_embed = model_out[pos_batch]
        new_dists = F.logsigmoid(F.cosine_similarity(model_out, pos_embed))
        if pos_dists is None:
            pos_dists = new_dists
        elif FLAGS.pos_neg_agg_method == 'min_max':
            pos_dists = torch.minimum(new_dists, pos_dists)
        else:
            pos_dists += new_dists

    if FLAGS.pos_neg_agg_method == 'mean':
        pos_dists /= FLAGS.pos_samples
        neg_dists /= FLAGS.neg_samples

    return torch.mean(torch.clamp(neg_dists - pos_dists + FLAGS.margin, min=0))


def perform_transductive_margin_training(
    data, edge_split, output_dir, device, input_size: int, has_features: bool, g_zoo
):
    """Trains ML-GCN on the transductive data and returns the trained model.
    Also returns timing information.
    """
    valid_edge = edge_split['valid']['edge'].T.to(device)

    model = g_zoo.get_model(
        FLAGS.graph_encoder_model,
        input_size,
        has_features,
        data.num_nodes,
        batched=False,
        n_feats=data.x.size(1),
    ).to(device)
    n_nodes = data.num_nodes

    adj = SparseTensor(
        row=data.edge_index[0],
        col=data.edge_index[1],
        value=None,
        sparse_sizes=(data.num_nodes, data.num_nodes),
    )
    adj_t = adj.t()
    row, col, _ = adj_t.coo()

    val_adj = SparseTensor(
        row=valid_edge[0],
        col=valid_edge[1],
        value=None,
        sparse_sizes=(data.num_nodes, data.num_nodes),
    )
    val_adj_t = val_adj.t()
    val_row, val_col, _ = val_adj_t.coo()

    optimizer = AdamW(
        list(model.parameters()), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)

    #####
    # Train & eval functions
    #####

    def train(step):
        model.train()
        optimizer.zero_grad()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if not has_features:
            data.x = model.get_node_feats().weight.data.clone().detach()

        model_out = model(data)
        loss = compute_margin_loss(
            device, n_nodes, data.edge_index, row, col, model_out
        )

        loss.backward()
        optimizer.step()

        wandb.log(
            {'curr_lr': lr, 'train_loss': loss, 'step': step, 'epoch': step}, step=step
        )
        return loss

    @torch.no_grad()
    def eval(step):
        model.eval()
        model_out = model.split_forward(data.x, valid_edge)

        val_loss = compute_margin_loss(
            device, n_nodes, valid_edge, val_row, val_col, model_out
        )
        wandb.log({'margin_val_loss': val_loss, 'step': step, 'epoch': step}, step=step)

        return val_loss

    #####
    ## End train & eval functions
    #####

    DELTA = 0.0001
    PATIENCE = 25
    WARMUP_PERIOD = FLAGS.lr_warmup_epochs
    lowest_loss = None
    last_epoch = None

    times = []
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        st_time = time.time_ns()
        train(epoch - 1)
        elapsed = time.time_ns() - st_time
        times.append(elapsed)
        val_loss = float(eval(epoch - 1))

        if lowest_loss is None or (lowest_loss - val_loss > DELTA):
            lowest_loss = val_loss
            last_epoch = epoch
        if epoch > WARMUP_PERIOD and epoch - last_epoch > PATIENCE:  # type: ignore
            log.info(
                f'Stopping early, no improvement in training loss for {PATIENCE} epochs'
            )
            break

    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save(
        {'model': model.state_dict()},
        os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}.pt'),
    )
    model = model.eval()
    representations = compute_data_representations_only(
        model, data, device, has_features=has_features
    )
    torch.save(
        representations, os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}-repr.pt')
    )

    return model, representations, time_bundle


def perform_inductive_margin_training(
    train_data,
    val_data,
    data,
    output_dir,
    device,
    input_size: int,
    has_features: bool,
    g_zoo,
):
    """Trains the ML-GCN on the inductive data and returns the trained model.
    Also returns timing information.
    """

    train_data = train_data.to(device)

    training_edge = train_data.edge_index
    valid_edge = val_data.edge_index.to(device)

    train_nodes = train_data.num_nodes
    model = g_zoo.get_model(
        FLAGS.graph_encoder_model,
        input_size,
        has_features,
        train_nodes,
        batched=False,
        n_feats=data.x.size(1),
    ).to(device)

    adj = SparseTensor(
        row=training_edge[0],
        col=training_edge[1],
        value=None,
        sparse_sizes=(train_data.num_nodes, train_data.num_nodes),
    )
    adj_t = adj.t()
    row, col, _ = adj_t.coo()

    val_adj = SparseTensor(
        row=valid_edge[0],
        col=valid_edge[1],
        value=None,
        sparse_sizes=(train_data.num_nodes, train_data.num_nodes),
    )
    val_adj_t = val_adj.t()
    val_row, val_col, _ = val_adj_t.coo()

    # optimizer
    optimizer = AdamW(
        list(model.parameters()), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def train(step):
        model.train()
        optimizer.zero_grad()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if not has_features:
            raise NotImplementedError()

        model_out = model(train_data)
        sample_size = train_nodes * 2
        loss = compute_margin_loss(
            device, train_nodes, train_data.edge_index, row, col, model_out
        )

        loss.backward()
        optimizer.step()

        # log scalars
        wandb.log(
            {'curr_lr': lr, 'train_loss': loss, 'step': step, 'epoch': step}, step=step
        )

        return loss

    @torch.no_grad()
    def eval(step):
        model.eval()
        model_out = model.split_forward(train_data.x, valid_edge)

        val_loss = compute_margin_loss(
            device, train_nodes, valid_edge, val_row, val_col, model_out
        )
        wandb.log({'margin_val_loss': val_loss, 'step': step, 'epoch': step}, step=step)

        return val_loss

    #####
    ## End train & eval functions
    #####

    DELTA = 0.0001
    PATIENCE = 25
    WARMUP_PERIOD = FLAGS.lr_warmup_epochs
    lowest_loss = None
    last_epoch = None

    times = []
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        st_time = time.time_ns()
        train(epoch - 1)
        elapsed = time.time_ns() - st_time
        times.append(elapsed)
        val_loss = float(eval(epoch - 1))

        if lowest_loss is None or (lowest_loss - val_loss > DELTA):
            lowest_loss = val_loss
            last_epoch = epoch
        if epoch > WARMUP_PERIOD and epoch - last_epoch > PATIENCE:  # type: ignore
            log.info(
                f'Stopping early, no improvement in training loss for {PATIENCE} epochs'
            )
            break

    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save(
        {'model': model.state_dict()},
        os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}.pt'),
    )
    model = model.eval()
    representations = compute_data_representations_only(
        model, train_data, device, has_features=has_features
    )
    torch.save(
        representations, os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}-repr.pt')
    )

    return model, representations, time_bundle
