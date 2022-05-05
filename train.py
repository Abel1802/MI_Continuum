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
from utils import prepare_device
from data_loader import CPCDataset_sameSeq
from data_loader import ContinuumDateset
from scheduler import WarmupScheduler
from model import Encoder_f0
from model import Encoder
from model import Decoder
from model import CLUB
from model import MINE
from trainer import Trainer


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
    data_set = ContinuumDateset(spec_dir=f"{config['data_dir']}/mels",
                                f0_dir=f"{config['data_dir']}/f0")
    data_loader = DataLoader(data_set, config['train']['batch_size'], shuffle=True)
    # Build model architecture
    encoder_f0 = Encoder_f0(config['model']['emb_lf0'])
    encoder = Encoder()
    decoder = Decoder()
    if config['MINE_net']:
        pm_mi_net = MINE(config['model']['lf0_size'], 
                        config['model']['z_dim'],
                        config['model']['hidden_size'])
    else:
        pm_mi_net = CLUB(config['model']['lf0_size'], 
                        config['model']['z_dim'],
                        config['model']['hidden_size'])
    models = [pm_mi_net, encoder_f0, encoder, decoder]

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    if len(device_ids) > 1:
        encoder_f0 = torch.nn.DataParallel(encoder_f0.cuda(), device_ids=device_ids)
        encoder = torch.nn.DataParallel(encoder.cuda(), device_ids=device_ids)
        decoder = torch.nn.DataParallel(decoder.cuda(), device_ids=device_ids)
        pm_mi_net = torch.nn.DataParallel(pm_mi_net.cuda(), device_ids=device_ids)
    else:
        encoder_f0 = encoder_f0.to(device)
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        pm_mi_net = pm_mi_net.to(device)
    logger.info(f"Prepare device on {device}, device_ids: {device_ids}")
    logger.info(f"training config: {config}")

    # Build optimizer, learning_rate scheduler
    optimizer = torch.optim.Adam(
        chain(encoder_f0.parameters(), encoder.parameters(), decoder.parameters()), 
        lr=config['scheduler']['initial_lr'])
    # warm-up
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=config['scheduler']['warmup_epochs'],
        initial_lr=config['scheduler']['initial_lr'],
        max_lr=config['scheduler']['max_lr'],
        milestones=config['scheduler']['milestones'],
        gamma=config['scheduler']['gamma'])
    optimizer_pm_mi_net = torch.optim.Adam(pm_mi_net.parameters(), 
                                           lr=config['train']['mi_lr'])
    optimizers = [optimizer_pm_mi_net, optimizer]

    trainer = Trainer(models, optimizers,
                      config=config,
                      logger=logger,
                      data_loader=data_loader,
                      device=device)
    
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='MI_Continuum')
    args.add_argument('-c', '--config', default="config.yaml", type=str,
                      help='config file path (default: None)')
    
    args = args.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)