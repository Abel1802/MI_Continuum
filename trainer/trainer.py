import numpy as np
import torch
import torch.nn.functional as F
import os
import time

from utils import to_train, to_eval


class Trainer():
    '''
        Trainer class
    '''
    def __init__(self, models, optimizers, config, logger,
                 data_loader, device, lr_schedule=None, len_epoch=None):
        super(Trainer, self).__init__()
        self.mi_pm_net, self.encoder_f0, self.encoder, self.decoder = models
        self.optimizer_pm_mi, self.optimizer = optimizers
        self.config = config
        self.logger = logger
        self.data_loader = data_loader
        # self.scheduler = scheduler
        self.device = device
        self.weight_pm_mi = config['weight_pm_mi']
        self.start_epoch = 1
        self.global_step = 1
        self.save_period = config['train']['save_period']
        self.epochs = config['train']['epochs']
        self.checkpoint_dir = f"exp/{config['exp_name']}/checkpoints"

    def _save_checkpoint(self, epoch):
        self.logger.info(f"Start to save checkpoint for epoch {epoch}...")
        checkpoint_state = {
            'encoder_f0': self.encoder_f0.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'mi_pm_net': self.mi_pm_net.state_dict(),
            'epoch': epoch
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = f"{self.checkpoint_dir}/model.ckpt-{epoch}.pt"
        torch.save(checkpoint_state, checkpoint_path)
        self.logger.info(f"Saved checkpoint {checkpoint_path}")
    
    def train(self):
        '''
            Full training logic
        '''
        self.logger.info("Strat training!")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

            if epoch % self.save_period == 0:
                # self._eval_model(epoch)
                self._save_checkpoint(epoch)
        self.logger.info("Training finished!")

    def mi_first(self, mels, lf0s):
        self.optimizer_pm_mi.zero_grad()
        z = self.encoder(mels).detach()
        lf0_embs = self.encoder_f0(lf0s).detach()
        lld_pm_loss = self.mi_pm_net.learning_loss(lf0_embs, z)
        lld_pm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mi_pm_net.parameters(), 1)
        self.optimizer_pm_mi.step()
        return lld_pm_loss
    
    def _eval_model(self, epoch):
        self.logger.info(f"Strat to eval model in epoch{epoch}...")
        average_rec_loss = average_lld_pm_loss = average_pm_mi_loss = average_pm_mi = 0
        stime = time.time()
        all_models = [self.encoder_f0, self.encoder, self.decoder, self.mi_pm_net]
        to_eval(all_models)
        for batch_idx, (mels, lf0s) in enumerate(self.valid_data_loader, 1):
            mels, lf0s = mels.float().to(self.device), lf0s.float().to(self.device)
     
            with torch.no_grad():
                z = self.encoder(mels)
                lf0_embs = self.encoder_f0(lf0s)
                mels_pred, mels_pred_postnet = self.decoder(z, lf0_embs)

                lld_pm_loss = self.mi_pm_net.learning_loss(lf0_embs, z)
                rec_loss = self.decoder.loss_function(mels, mels_pred, mels_pred_postnet)
                pm_mi = self.mi_pm_net(lf0_embs, z)
                pm_mi_loss = self.weight_pm_mi * pm_mi
                loss = rec_loss + pm_mi_loss

            average_rec_loss += (rec_loss.item() - average_rec_loss) / batch_idx
            average_lld_pm_loss += (lld_pm_loss.item() - average_lld_pm_loss) / batch_idx
            average_pm_mi_loss += (pm_mi_loss.item() - average_pm_mi_loss) / batch_idx
            average_pm_mi += (pm_mi.item() - average_pm_mi) / batch_idx

        ctime = time.time()
        self.logger.info(f"Eval valid_set Epoch: {epoch} rec_loss: {average_rec_loss:.6f} "
                        f"lld_pm_loss: {average_lld_pm_loss:.6f} pm_mi_loss: {average_pm_mi_loss:.6f} used_time: {ctime-stime:.3f}s")
        to_train(all_models)
        
    def _train_epoch(self, epoch):
        '''
            Training logic for an epoch

            :param epoch: Integer, current training epoch.
            :return A log that contains average loss and metric in this epoch.
        '''
        average_rec_loss = average_lld_pm_loss = average_pm_mi_loss = average_pm_mi = 0
        stime = time.time()
        for batch_idx, (mels, lf0s) in enumerate(self.data_loader, 1):
            mels, lf0s = mels.float().to(self.device), lf0s.float().to(self.device)
            # mi_first
            for j in range(5):
                lld_pm_loss = self.mi_first(mels, lf0s)

            # mi_second
            self.optimizer.zero_grad()
            z = self.encoder(mels)
            lf0_embs = self.encoder_f0(lf0s)
            
            mels_pred, mels_pred_postnet = self.decoder(z, lf0_embs)
            rec_loss = self.decoder.loss_function(mels, mels_pred, mels_pred_postnet)
            pm_mi = self.mi_pm_net(lf0_embs, z)
            pm_mi_loss = self.weight_pm_mi * pm_mi
            loss = rec_loss + pm_mi_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            average_rec_loss += (rec_loss.item() - average_rec_loss) / batch_idx
            average_lld_pm_loss += (lld_pm_loss.item() - average_lld_pm_loss) / batch_idx
            average_pm_mi_loss += (pm_mi_loss.item() - average_pm_mi_loss) / batch_idx
            average_pm_mi += (pm_mi.item() - average_pm_mi) / batch_idx

            ctime = time.time()
            print(f"Train Epoch: {epoch} global_step: {self.global_step} rec_loss: {average_rec_loss:.6f} " 
                  f"lld_pm_loss: {average_lld_pm_loss:.6f} pm_mi_loss: {average_pm_mi_loss:.6f} "
                  f"used_time: {ctime-stime:.3f}s")
            self.global_step += 1
            stime = time.time()
        # self.scheduler.step()
        # lr = self.scheduler.get_last_lr()
        # self.logger.info(f"lr: {lr}")
        self.logger.info(f"Train Epoch: {epoch} global_step: {self.global_step} rec_loss: {average_rec_loss:.6f} "
                             f"lld_pm_loss: {average_lld_pm_loss:.6f} pm_mi_loss: {average_pm_mi_loss:.6f}")