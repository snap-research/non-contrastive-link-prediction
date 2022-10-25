import copy
import os
import time
import torch
from absl import flags
from torch import cosine_similarity, nn
from torch.optim import AdamW
from torch_cluster import random_walk
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch_geometric.utils import negative_sampling
import wandb
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
import logging
from torch.nn import TripletMarginWithDistanceLoss
from torch.optim.lr_scheduler import CyclicLR
import numpy as np

from .loss import barlow_twins_loss, cca_ssg_loss
from .ncl import CCASSG, GraphBarlowTwins, SimSiam
from .scheduler import CosineDecayScheduler
from .bgrl import BGRL, TripletBGRL, compute_data_representations_only, compute_representations_only
from .transforms import compose_transforms
from .predictors import MLP_Predictor
from torch_geometric.nn import SAGEConv

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_time_bundle(times):
    times = np.array(times)
    std_time, mean_time = np.std(times), np.mean(times)
    total_time = np.sum(times)
    return (total_time, std_time, mean_time, times)


class SAGE(nn.Module):

    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def bdot(a, b):
    return torch.bmm(a.view(a.size(0), 1, -1), b.view(b.size(0), -1, 1))


def edges2dict(edges, n_nodes, n_samples):
    out = torch.ones((n_nodes, n_samples), dtype=torch.long, device=edges.device) * -1
    for edge in edges.T:
        l, r = edge
        for k in range(n_samples):
            if torch.isnan(out[l, k]):
                out[l, k] = r
                break

    is_nan = (out == -1)
    to_gen = is_nan.sum()
    # fill the ones we couldn't find negative edges for
    out[is_nan] = torch.randint(0, n_nodes, (to_gen,), dtype=torch.long, device=out.device)
    return out


def compute_margin_loss(device, n_nodes, edge_index, row, col, model_out):
    neg_dists = None
    for _ in range(FLAGS.neg_samples):
        neg_edges = negative_sampling(edge_index, n_nodes, n_nodes)
        lookup = edges2dict(neg_edges, n_nodes, 1)

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
        pos_batch = random_walk(row, col, node_vec, walk_length=1, coalesced=False)[:, 1]
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


def perform_inductive_margin_training(train_data, val_data, inference_data, data, test_edge_bundle, negative_samples,
                                      output_dir, writer, device, input_size: int, has_features: bool, g_zoo):
    train_data = train_data.to(device)

    training_edge = train_data.edge_index
    valid_edge = val_data.edge_index.to(device)

    train_nodes = train_data.num_nodes
    model = g_zoo.get_model(FLAGS.graph_encoder_model,
                            input_size,
                            has_features,
                            train_nodes,
                            batched=False,
                            n_feats=data.x.size(1)).to(device)

    adj = SparseTensor(row=training_edge[0],
                       col=training_edge[1],
                       value=None,
                       sparse_sizes=(train_data.num_nodes, train_data.num_nodes))
    adj_t = adj.t()
    row, col, _ = adj_t.coo()

    val_adj = SparseTensor(row=valid_edge[0],
                           col=valid_edge[1],
                           value=None,
                           sparse_sizes=(train_data.num_nodes, train_data.num_nodes))
    val_adj_t = val_adj.t()
    val_row, val_col, _ = val_adj_t.coo()

    # optimizer
    optimizer = AdamW(list(model.parameters()), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    # Store it on CPU memory to save GPU memory for certain sampling operations

    def train(step):
        model.train()
        optimizer.zero_grad()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        if not has_features:
            raise NotImplementedError()

        model_out = model(train_data)
        sample_size = train_nodes * 2
        loss = compute_margin_loss(device, train_nodes, train_data.edge_index, row, col, model_out)

        loss.backward()
        optimizer.step()

        # log scalars
        wandb.log({'curr_lr': lr, 'curr_mm': mm, 'train_loss': loss, 'step': step, 'epoch': step}, step=step)

        writer.add_scalar('gcn/params/lr', lr, step)
        writer.add_scalar('gcn/params/mm', mm, step)
        writer.add_scalar('gcn/train/loss', loss, step)
        return loss

    @torch.no_grad()
    def eval(step):
        model.eval()
        model_out = model.split_forward(train_data.x, valid_edge)

        val_loss = compute_margin_loss(device, train_nodes, valid_edge, val_row, val_col, model_out)
        wandb.log({'margin_val_loss': val_loss, 'step': step, 'epoch': step}, step=step)

        writer.add_scalar('gcn/val/loss', val_loss, step)
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
            log.info(f'Stopping early, no improvement in training loss for {PATIENCE} epochs')
            break

    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save({'model': model.state_dict()}, os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}.pt'))
    model = model.eval()
    # with torch.no_grad():
    #     representations = (model.full_forward(data.x, data.edge_index))
    representations = compute_data_representations_only(model, train_data, device, has_features=has_features)
    torch.save(representations, os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}-repr.pt'))

    return model, representations, time_bundle


def perform_gcn_margin_training(data, edge_split, output_dir, writer, device, input_size: int, has_features: bool,
                                g_zoo):
    valid_edge = edge_split['valid']['edge'].T.to(device)

    model = g_zoo.get_model(FLAGS.graph_encoder_model,
                            input_size,
                            has_features,
                            data.num_nodes,
                            batched=False,
                            n_feats=data.x.size(1)).to(device)
    n_nodes = data.num_nodes

    adj = SparseTensor(row=data.edge_index[0],
                       col=data.edge_index[1],
                       value=None,
                       sparse_sizes=(data.num_nodes, data.num_nodes))
    adj_t = adj.t()
    row, col, _ = adj_t.coo()

    val_adj = SparseTensor(row=valid_edge[0],
                           col=valid_edge[1],
                           value=None,
                           sparse_sizes=(data.num_nodes, data.num_nodes))
    val_adj_t = val_adj.t()
    val_row, val_col, _ = val_adj_t.coo()

    # optimizer
    optimizer = AdamW(list(model.parameters()), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    # we already filtered out test/val edges
    train_edge = data.edge_index

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

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        if not has_features:
            data.x = model.get_node_feats().weight.data.clone().detach()

        model_out = model(data)
        embeddings = model_out

        sample_size = n_nodes * 2
        loss = compute_margin_loss(device, n_nodes, data.edge_index, row, col, model_out)

        loss.backward()
        optimizer.step()

        # log scalars
        wandb.log({'curr_lr': lr, 'curr_mm': mm, 'train_loss': loss, 'step': step, 'epoch': step}, step=step)

        writer.add_scalar('gcn/params/lr', lr, step)
        writer.add_scalar('gcn/params/mm', mm, step)
        writer.add_scalar('gcn/train/loss', loss, step)
        return loss

    @torch.no_grad()
    def eval(step):
        model.eval()
        model_out = model.split_forward(data.x, valid_edge)
        # embeddings = nn.Embedding.from_pretrained(model_out, freeze=True)

        val_loss = compute_margin_loss(device, n_nodes, valid_edge, val_row, val_col, model_out)
        wandb.log({'margin_val_loss': val_loss, 'step': step, 'epoch': step}, step=step)

        writer.add_scalar('gcn/val/loss', val_loss, step)
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
            log.info(f'Stopping early, no improvement in training loss for {PATIENCE} epochs')
            break

    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save({'model': model.state_dict()}, os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}.pt'))
    model = model.eval()
    representations = compute_data_representations_only(model, data, device, has_features=has_features)
    torch.save(representations, os.path.join(output_dir, f'gcn-ml-{FLAGS.dataset}-repr.pt'))

    return model, representations, time_bundle


def perform_gbt_training(data, output_dir, device, input_size: int, has_features: bool, g_zoo):
    # prepare transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_1,
                                     drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_2,
                                     drop_feat_p=FLAGS.drop_feat_p_2)

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model,
                              input_size,
                              has_features,
                              data.num_nodes,
                              n_feats=data.x.size(1))
    # predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = GraphBarlowTwins(encoder, has_features=has_features).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def full_train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # forward
        optimizer.zero_grad()

        if not has_features:
            data.x = encoder.get_node_feats().weight.data.clone().detach()

        x1, x2 = transform_1(data), transform_2(data)

        y1, y2 = model(x1, x2)

        loss = barlow_twins_loss(y1, y2)
        loss.backward()

        # update online network
        optimizer.step()

        # log scalars
        wandb.log({'curr_lr': lr, 'train_loss': loss, 'step': step, 'epoch': epoch}, step=step)

    times = []
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        st_time = time.time_ns()
        full_train(epoch - 1)
        elapsed = time.time_ns() - st_time
        times.append(elapsed)
    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save({'model': model.encoder.state_dict()}, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'))
    encoder = copy.deepcopy(model.encoder.eval())
    representations = compute_data_representations_only(encoder, data, device, has_features=has_features)
    torch.save(representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt'))

    return encoder, representations, time_bundle


def perform_cca_ssg_training(data, output_dir, device, input_size: int, has_features: bool, g_zoo):
    # prepare transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_1,
                                     drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_2,
                                     drop_feat_p=FLAGS.drop_feat_p_2)

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model,
                              input_size,
                              has_features,
                              data.num_nodes,
                              n_feats=data.x.size(1))
    model = CCASSG(encoder, has_features=has_features).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def full_train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # forward
        optimizer.zero_grad()

        if not has_features:
            data.x = encoder.get_node_feats().weight.data.clone().detach()

        x1, x2 = transform_1(data), transform_2(data)

        y1, y2 = model(x1, x2)

        loss = cca_ssg_loss(y1, y2, FLAGS.cca_lambda, data.num_nodes)
        loss.backward()

        # update online network
        optimizer.step()

        # log scalars
        wandb.log({'curr_lr': lr, 'train_loss': loss, 'step': step, 'epoch': epoch}, step=step)

    times = []
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        st_time = time.time_ns()
        full_train(epoch - 1)
        elapsed = time.time_ns() - st_time
        times.append(elapsed)
    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save({'model': model.encoder.state_dict()}, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'))
    encoder = copy.deepcopy(model.encoder.eval())
    representations = compute_data_representations_only(encoder, data, device, has_features=has_features)
    torch.save(representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt'))

    return encoder, representations, time_bundle


def perform_simsiam_training(data,
                             output_dir,
                             representation_size,
                             device,
                             input_size: int,
                             has_features: bool,
                             g_zoo,
                             train_cb=None):
    # prepare transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_1,
                                     drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_2,
                                     drop_feat_p=FLAGS.drop_feat_p_2)

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model,
                              input_size,
                              has_features,
                              data.num_nodes,
                              n_feats=data.x.size(1))
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = SimSiam(encoder, predictor, has_features=has_features).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def full_train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # forward
        optimizer.zero_grad()

        if not has_features:
            data.x = encoder.get_node_feats().weight.data.clone().detach()
        x1, x2 = transform_1(data), transform_2(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)

        loss = 2 - F.cosine_similarity(q1, y2.detach()).mean() - F.cosine_similarity(q2, y1.detach()).mean()
        loss.backward()

        # update online network
        optimizer.step()

        # log scalars
        wandb.log({'curr_lr': lr, 'train_loss': loss, 'step': step, 'epoch': epoch}, step=step)

    # training loop
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        if train_cb is not None:
            train_cb(epoch - 1, model)
        full_train(epoch - 1)

    # save encoder weights
    torch.save({'model': model.encoder.state_dict()}, os.path.join(output_dir, f'simsiam-{FLAGS.dataset}.pt'))
    encoder = copy.deepcopy(model.encoder.eval())
    representations = compute_data_representations_only(encoder, data, device, has_features=has_features)
    torch.save(representations, os.path.join(output_dir, f'simsiam-{FLAGS.dataset}-repr.pt'))

    return encoder, representations


def cosine_distance(A, B):
    sims = cosine_similarity(A, B)
    return -(sims + 1) / 2 + 1


def perform_triplet_training(data,
                             output_dir,
                             representation_size,
                             device,
                             input_size: int,
                             has_features: bool,
                             g_zoo,
                             train_cb=None):
    # prepare transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_1,
                                     drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_2,
                                     drop_feat_p=FLAGS.drop_feat_p_2)
    transform_3 = compose_transforms(FLAGS.negative_transforms, drop_edge_p=0.95, drop_feat_p=0.95)

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model,
                              input_size,
                              has_features,
                              data.num_nodes,
                              n_feats=data.x.size(1))
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = TripletBGRL(encoder, predictor, has_features=has_features).to(device)
    neg_lambda = FLAGS.neg_lambda

    # optimizer
    # fake_opt = SGD(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    if FLAGS.scheduler == 'cyclic':
        lr_scheduler = CyclicLR(optimizer, FLAGS.lr, FLAGS.cyclic_lr, step_size_up=100, cycle_momentum=False)
    else:
        lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    margin_loss = TripletMarginWithDistanceLoss(margin=FLAGS.margin,
                                                reduction='mean',
                                                distance_function=cosine_distance)

    #####
    # Train & eval functions
    #####
    def full_train(step):
        model.train()

        # update learning rate
        if FLAGS.scheduler == 'cyclic':
            lr = lr_scheduler.get_last_lr()[0]
        else:
            lr = lr_scheduler.get(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        if not has_features:
            data.x = encoder.get_node_feats().weight.data.clone().detach()
        x1, x2, x3 = transform_1(data), transform_2(data), transform_3(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        neg_y = model.forward_target(x3)

        if FLAGS.hybrid_loss and step < FLAGS.hybrid_transition_epoch:
            loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(),
                                                                                             dim=-1).mean()
            to_log = dict()
        else:
            if FLAGS.use_margin:
                loss_1 = margin_loss(q1, y2, neg_y)
                loss_2 = margin_loss(q2, y1, neg_y)
                to_log = {'loss_1': loss_1, 'loss_2': loss_2}
                loss = (loss_1 + loss_2) / 2
            else:
                sim1 = F.cosine_similarity(q1, y2.detach()).mean()
                sim2 = F.cosine_similarity(q2, y1.detach()).mean()
                neg_sim1 = F.cosine_similarity(q1, neg_y.detach()).mean()
                neg_sim2 = F.cosine_similarity(q2, neg_y.detach()).mean()
                to_log = {'sim1': sim1, 'sim2': sim2, 'neg_sim1': neg_sim1, 'neg_sim2': neg_sim2}

                loss = neg_lambda * (neg_sim1 + neg_sim2) - (1 - neg_lambda) * (sim1 + sim2)

        loss.backward()

        # update online network

        optimizer.step()
        if FLAGS.scheduler == 'cyclic':
            lr_scheduler.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        wandb.log({'curr_lr': lr, 'curr_mm': mm, 'train_loss': loss, 'step': step, 'epoch': epoch, **to_log}, step=step)
        return loss

    best_loss = None
    last_update_epoch = 0
    times = []

    # training loop
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        if train_cb is not None:
            train_cb(epoch - 1, model)
        st_time = time.time_ns()

        train_loss = full_train(epoch - 1)

        elapsed = time.time_ns() - st_time
        times.append(elapsed)

        if best_loss is None or (best_loss - train_loss >= 0.01):
            best_loss = train_loss
            last_update_epoch = epoch
        elif FLAGS.training_early_stop and epoch - last_update_epoch > FLAGS.training_early_stop_patience:
            print('Early stopping performed!')
            break
    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(output_dir, f'triplet-{FLAGS.dataset}.pt'))
    encoder = copy.deepcopy(model.online_encoder.eval())
    representations = compute_data_representations_only(encoder, data, device, has_features=has_features)
    torch.save(representations, os.path.join(output_dir, f'triplet-{FLAGS.dataset}-repr.pt'))

    return encoder, representations, time_bundle


def perform_bgrl_training(data,
                          output_dir,
                          representation_size,
                          device,
                          input_size: int,
                          has_features: bool,
                          g_zoo,
                          dataset=None,
                          num_eval_splits=None,
                          train_cb=None,
                          extra_return=False):
    # prepare transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_1,
                                     drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_2,
                                     drop_feat_p=FLAGS.drop_feat_p_2)

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model,
                              input_size,
                              has_features,
                              data.num_nodes,
                              n_feats=data.x.size(1))
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor, has_features=has_features).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def full_train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        if not has_features:
            data.x = encoder.get_node_feats().weight.data.clone().detach()
        x1, x2 = transform_1(data), transform_2(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)

        loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        wandb.log({'curr_lr': lr, 'curr_mm': mm, 'train_loss': loss, 'step': step, 'epoch': epoch}, step=step)

    def batch_train(loader, epoch):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(epoch)

        # TODO(author): add support for keeping feature matrix on CPU if graph is too large
        for batch in tqdm(iterable=loader, desc='Batches', leave=False):
            batch = batch.to(device)
            # adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()

            if not has_features:
                raise NotImplementedError()

            x1, x2 = transform_1(batch), transform_2(batch)

            q1, y2 = model(x1, x2)
            q2, y1 = model(x2, x1)

            loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(),
                                                                                             dim=-1).mean()
            loss.backward()

        optimizer.step()
        # update target network
        model.update_target_network(mm)

    #####
    ## End train & eval functions
    #####
    if FLAGS.batch_graphs:
        times = []
        train_loader = NeighborLoader(
            data,
            # sizes from https://github.com/pbielak/graph-barlow-twins/blob/master/experiments/scripts/batched/hps_bgrl.py
            num_neighbors=[
                10,
            ] * encoder.num_layers,
            batch_size=FLAGS.graph_batch_size,
            shuffle=True,
            num_workers=FLAGS.n_workers)

        for epoch in tqdm(range(1, FLAGS.epochs + 1)):
            batch_train(train_loader, epoch - 1)
    else:
        data = data.to(device)
        times = []

        for epoch in tqdm(range(1, FLAGS.epochs + 1)):
            if train_cb is not None:
                train_cb(epoch - 1, model)
            st_time = time.time_ns()
            full_train(epoch - 1)
            elapsed = time.time_ns() - st_time
            times.append(elapsed)

    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'))
    encoder = copy.deepcopy(model.online_encoder.eval())

    # if FLAGS.batch_graphs:
    #     # TODO(author): make sure we need to convert embeddings at this stage,
    #     #  may be worth saving memory by keeping on CPU for very large graphs
    #     representations = compute_batch_data_representations(encoder, data, device).to(device)
    # else:
    representations = compute_data_representations_only(encoder, data, device, has_features=has_features)
    torch.save(representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt'))

    if extra_return:
        return encoder, predictor

    return encoder, representations, time_bundle


def perform_inductive_bgrl_training(train_dataset,
                                    train_loader,
                                    output_dir,
                                    representation_size,
                                    device,
                                    g_zoo,
                                    train_cb=None):
    # prepare transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_1,
                                     drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_2,
                                     drop_feat_p=FLAGS.drop_feat_p_2)
    input_size = train_dataset.num_node_features

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, True, 0, n_feats=input_size)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor, has_features=True).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def full_train(data, step):
        model.train()
        data = data.to(device)

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        x1, x2 = transform_1(data), transform_2(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)

        loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        wandb.log({'curr_lr': lr, 'curr_mm': mm, 'train_loss': loss, 'step': step, 'epoch': epoch}, step=step)

        # log scalars

    #####
    ## End train & eval functions
    #####

    train_iter = iter(train_loader)
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        if train_cb is not None:
            train_cb(epoch - 1, model)

        data = next(train_iter, None)
        if data is None:
            train_iter = iter(train_loader)
            data = next(train_iter, None)
        full_train(data, epoch)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'))
    encoder = copy.deepcopy(model.online_encoder.eval())

    # if FLAGS.batch_graphs:
    #     # TODO(author): make sure we need to convert embeddings at this stage,
    #     #  may be worth saving memory by keeping on CPU for very large graphs
    #     representations = compute_batch_data_representations(encoder, data, device).to(device)
    # else:
    representations = compute_representations_only(encoder, train_dataset, device, has_features=True)
    torch.save(representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt'))

    return encoder, representations


def perform_triplet_bgrl_training(train_dataset,
                                  train_loader,
                                  output_dir,
                                  representation_size,
                                  device,
                                  g_zoo,
                                  train_cb=None):
    # prepare transforms
    transform_1 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_1,
                                     drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = compose_transforms(FLAGS.graph_transforms,
                                     drop_edge_p=FLAGS.drop_edge_p_2,
                                     drop_feat_p=FLAGS.drop_feat_p_2)
    transform_3 = compose_transforms(FLAGS.negative_transforms, drop_edge_p=0.98, drop_feat_p=0.98)
    input_size = train_dataset.num_node_features

    encoder = g_zoo.get_model(FLAGS.graph_encoder_model, input_size, True, 0, n_feats=input_size)
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    model = TripletBGRL(encoder, predictor, has_features=True).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def full_train(data, step):
        model.train()
        data = data.to(device)

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        x1, x2, x3 = transform_1(data), transform_2(data), transform_3(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        neg_y = model.forward_target(x3)

        sim1 = F.cosine_similarity(q1, y2.detach()).mean()
        sim2 = F.cosine_similarity(q2, y1.detach()).mean()
        neg_sim1 = F.cosine_similarity(q1, neg_y.detach()).mean()
        neg_sim2 = F.cosine_similarity(q2, neg_y.detach()).mean()

        loss = neg_sim1 + neg_sim2 - sim1 - sim2
        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)

        # log scalars
        wandb.log(
            {
                'curr_lr': lr,
                'curr_mm': mm,
                'train_loss': loss,
                'step': step,
                'epoch': epoch,
                'sim1': sim1,
                'sim2': sim2,
                'neg_sim1': neg_sim1,
                'neg_sim2': neg_sim2
            },
            step=step)

        # log scalars

    #####
    ## End train & eval functions
    #####

    train_iter = iter(train_loader)
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        if train_cb is not None:
            train_cb(epoch - 1, model)

        data = next(train_iter, None)
        if data is None:
            train_iter = iter(train_loader)
            data = next(train_iter, None)
        full_train(data, epoch)

    # save encoder weights
    torch.save({'model': model.online_encoder.state_dict()}, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'))
    encoder = copy.deepcopy(model.online_encoder.eval())

    # if FLAGS.batch_graphs:
    #     # TODO(author): make sure we need to convert embeddings at this stage,
    #     #  may be worth saving memory by keeping on CPU for very large graphs
    #     representations = compute_batch_data_representations(encoder, data, device).to(device)
    # else:
    representations = compute_representations_only(encoder, train_dataset, device, has_features=True)
    torch.save(representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt'))

    return encoder, representations
