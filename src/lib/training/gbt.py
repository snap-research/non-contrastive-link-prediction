import copy
import os
import time
import torch
from absl import flags
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import logging

from .utils import get_time_bundle
from ..loss import barlow_twins_loss
from ..scheduler import CosineDecayScheduler
from ..utils import compute_data_representations_only
from ..transforms import compose_transforms

from ..models.gbt import GraphBarlowTwins

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def perform_gbt_training(
    data, output_dir, device, input_size: int, has_features: bool, enc_zoo
):
    """Train a Graph Barlow Twins model on the data.
    Works for both the transductive and inductive settings (only difference is the data passed in).
    """
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

    encoder = enc_zoo.get_model(
        FLAGS.graph_encoder_model,
        input_size,
        has_features,
        data.num_nodes,
        n_feats=data.x.size(1),
    )
    model = GraphBarlowTwins(encoder, has_features=has_features).to(device)

    optimizer = AdamW(
        model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )
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
            data.x = encoder.get_node_feats().weight.data

        x1, x2 = transform_1(data), transform_2(data)
        y1, y2 = model(x1, x2)

        loss = barlow_twins_loss(y1, y2)

        loss.backward()
        optimizer.step()

        wandb.log(
            {'curr_lr': lr, 'train_loss': loss, 'step': step, 'epoch': epoch}, step=step
        )

    times = []
    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        st_time = time.time_ns()
        full_train(epoch - 1)
        elapsed = time.time_ns() - st_time
        times.append(elapsed)
    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save(
        {'model': model.encoder.state_dict()},
        os.path.join(output_dir, f'bgrl-{FLAGS.dataset}.pt'),
    )
    encoder = copy.deepcopy(model.encoder.eval())
    representations = compute_data_representations_only(
        encoder, data, device, has_features=has_features
    )
    torch.save(
        representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt')
    )

    return encoder, representations, time_bundle
