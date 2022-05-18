import argparse
import yaml
import os
import hydra
from hydra import utils
from pathlib import Path
import logging
from itertools import chain
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import get_logger
from utils import try_gup
from data_loader import CPCDataset_sameSeq
from data_loader import ContinuumDateset
from data_loader import ContinuumMultiDateset
from scheduler import WarmupScheduler
from model import Encoder_f0
from model import Encoder
from model import DecoderMulti
from model import CLUB
from model import MINE
from trainer import TrainerMulti


# Fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    exp_dir = f"exp/{config['exp_name']}"
    os.makedirs(exp_dir, exist_ok=True)
    logger = get_logger(f'{exp_dir}/train.log')

    # Setup data_loader
    data_set = ContinuumMultiDateset(spec_dir=f"{config['data_dir']}/mels",
                                F1_dir=f"{config['data_dir']}/F1",
                                F2_dir=f"{config['data_dir']}/F2")
    data_loader = DataLoader(data_set, config['train']['batch_size'], shuffle=True)
    # Build model architecture
    encoder_F1 = Encoder_f0(config['model']['emb_lf0'])
    encoder_F2 = Encoder_f0(config['model']['emb_lf0'])
    encoder = Encoder()
    decoder = DecoderMulti()
    if config['MINE_net']:
        m1_mi_net = MINE(config['model']['lf0_size'], 
                        config['model']['z_dim'],
                        config['model']['hidden_size'])
        m2_mi_net = MINE(config['model']['lf0_size'], 
                        config['model']['z_dim'],
                        config['model']['hidden_size'])

    else:
        m1_mi_net = CLUB(config['model']['lf0_size'], 
                        config['model']['z_dim'],
                        config['model']['hidden_size'])
    models = [m1_mi_net, m2_mi_net, encoder_F1, encoder_F2, encoder, decoder]

    # prepare for (multi-device) GPU training
    device = try_gup(config['gpu_id'])
    encoder_F1 = encoder_F1.to(device)
    encoder_F2 = encoder_F2.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    m1_mi_net = m1_mi_net.to(device)
    m2_mi_net = m2_mi_net.to(device)
    logger.info(f"Prepare device on {device}")
    logger.info(f"training config: {config}")

    # Build optimizer, learning_rate scheduler
    optimizer = torch.optim.Adam(
        chain(encoder_F1.parameters(), encoder_F2.parameters(), encoder.parameters(), decoder.parameters()), 
        lr=config['scheduler']['initial_lr'])
    # warm-up
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=config['scheduler']['warmup_epochs'],
        initial_lr=config['scheduler']['initial_lr'],
        max_lr=config['scheduler']['max_lr'],
        milestones=config['scheduler']['milestones'],
        gamma=config['scheduler']['gamma'])
    optimizer_m1_mi_net = torch.optim.Adam(m1_mi_net.parameters(), 
                                           lr=config['train']['mi_lr'])
    optimizer_m2_mi_net = torch.optim.Adam(m2_mi_net.parameters(), 
                                           lr=config['train']['mi_lr'])

    optimizers = [optimizer_m1_mi_net, optimizer_m2_mi_net, optimizer]

    trainer = TrainerMulti(models, optimizers,
                      config=config,
                      logger=logger,
                      data_loader=data_loader,
                      device=device,
                      scheduler=scheduler)
    
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='MI_Continuum')
    args.add_argument('-c', '--config', default="config.yaml", type=str,
                      help='config file path (default: None)')
    
    args = args.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
