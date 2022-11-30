import copy
import os
import time
import torch
import logging
from absl import flags
from torch import cosine_similarity
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from torch_geometric.loader import NeighborLoader

from .utils import get_time_bundle
from ..scheduler import CosineDecayScheduler
from ..utils import compute_data_representations_only
from ..transforms import compose_transforms

from ..models.bgrl import BGRL, MlpPredictor

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def perform_bgrl_training(
    data,
    output_dir,
    representation_size,
    device,
    input_size: int,
    has_features: bool,
    g_zoo,
    dataset=None,
    train_cb=None,
    extra_return=False,
):
    """Trains Bootstrapped Representation Learning on Graphs (BGRL)."""

    # prepare transforms
    transform_1 = compose_transforms(
        FLAGS.graph_transforms,
        drop_edge_p=FLAGS.drop_edge_p_1,
        drop_feat_p=FLAGS.drop_feat_p_1,
    )
    transform_2 = compose_transforms(
        FLAGS.graph_transforms,
        drop_edge_p=FLAGS.drop_edge_p_2,
        drop_feat_p=FLAGS.drop_feat_p_2,
    )

    encoder = g_zoo.get_model(
        FLAGS.graph_encoder_model,
        input_size,
        has_features,
        data.num_nodes,
        n_feats=data.x.size(1),
    )
    predictor = MlpPredictor(
        representation_size,
        representation_size,
        hidden_size=FLAGS.predictor_hidden_size,
    )
    model = BGRL(encoder, predictor, has_features=has_features).to(device)

    # optimizer
    optimizer = AdamW(
        model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )

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

        optimizer.zero_grad()
        if not has_features:
            data.x = encoder.get_node_feats().weight.data.clone().detach()

        x1, x2 = transform_1(data), transform_2(data)
        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)

        loss = (
            2
            - cosine_similarity(q1, y2.detach(), dim=-1).mean()
            - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        )
        loss.backward()

        optimizer.step()
        model.update_target_network(mm)

        wandb.log(
            {
                'curr_lr': lr,
                'curr_mm': mm,
                'train_loss': loss,
                'step': step,
                'epoch': epoch,
            },
            step=step,
        )

    def batch_train(loader, epoch):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(epoch)

        for batch in tqdm(iterable=loader, desc='Batches', leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()

            if not has_features:
                raise NotImplementedError()

            x1, x2 = transform_1(batch), transform_2(batch)

            q1, y2 = model(x1, x2)
            q2, y1 = model(x2, x1)

            loss = (
                2
                - cosine_similarity(q1, y2.detach(), dim=-1).mean()
                - cosine_similarity(q2, y1.detach(), dim=-1).mean()
            )
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
            ]
            * encoder.num_layers,
            batch_size=FLAGS.graph_batch_size,
            shuffle=True,
            num_workers=FLAGS.n_workers,
        )

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
    torch.save(
        {'model': model.online_encoder.state_dict()},
        os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'),
    )
    encoder = copy.deepcopy(model.online_encoder.eval())

    representations = compute_data_representations_only(
        encoder, data, device, has_features=has_features
    )
    torch.save(
        representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt')
    )

    if extra_return:
        return encoder, predictor

    return encoder, representations, time_bundle
