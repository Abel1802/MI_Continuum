import numpy as np
import torch
import torch.nn.functional as F
import os
import time

from utils import to_train, to_eval


class TrainerMulti():
    '''
        Trainer class
    '''
    def __init__(self, models, optimizers, config, logger,
                 data_loader, device, scheduler, len_epoch=None):
        super(TrainerMulti, self).__init__()
        self.mi_m1_net, self.mi_m2_net, self.encoder_F1, self.encoder_F2, self.encoder, self.decoder = models
        self.optimizer_m1_mi, self.optimizer_m2_mi, self.optimizer = optimizers
        self.config = config
        self.logger = logger
        self.data_loader = data_loader
        self.scheduler = scheduler
        self.device = device
        self.weight_m1_mi = config['weight_m1_mi']
        self.start_epoch = 1
        self.global_step = 1
        self.save_period = config['train']['save_period']
        self.epochs = config['train']['epochs']
        self.checkpoint_dir = f"exp/{config['exp_name']}/checkpoints"

    def _save_checkpoint(self, epoch):
        self.logger.info(f"Start to save checkpoint for epoch {epoch}...")
        checkpoint_state = {
            'encoder_F1': self.encoder_F1.state_dict(),
            'encoder_F2': self.encoder_F2.state_dict(),
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'mi_m1_net': self.mi_m1_net.state_dict(),
            'mi_m2_net': self.mi_m2_net.state_dict(),
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

    def mi_first(self, mels, lF1s, lF2s):
        self.optimizer_m1_mi.zero_grad()
        self.optimizer_m2_mi.zero_grad()
        z = self.encoder(mels).detach()
        lF1_embs = self.encoder_F1(lF1s).detach()
        lF2_embs = self.encoder_F2(lF2s).detach()
        # MI between mel and F1
        lld_m1_loss = self.mi_m1_net.learning_loss(lF1_embs, z)
        lld_m1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mi_m1_net.parameters(), 1)
        self.optimizer_m1_mi.step()
        # MI between mel and F2
        lld_m2_loss = self.mi_m2_net.learning_loss(lF2_embs, z)
        lld_m2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mi_m2_net.parameters(), 1)
        self.optimizer_m2_mi.step()
        return lld_m1_loss, lld_m2_loss
    
    def _eval_model(self, epoch):
        self.logger.info(f"Strat to eval model in epoch{epoch}...")
        average_rec_loss = average_lld_m1_loss = average_m1_mi_loss = average_m1_mi = 0
        stime = time.time()
        all_models = [self.encoder_F1, self.encoder, self.decoder, self.mi_m1_net]
        to_eval(all_models)
        for batch_idx, (mels, lF1s) in enumerate(self.valid_data_loader, 1):
            mels, lF1s = mels.float().to(self.device), lF1s.float().to(self.device)
     
            with torch.no_grad():
                z = self.encoder(mels)
                lF1_embs = self.encoder_F1(lF1s)
                mels_pred, mels_pred_postnet = self.decoder(z, lF1_embs)

                lld_m1_loss = self.mi_m1_net.learning_loss(lF1_embs, z)
                rec_loss = self.decoder.loss_function(mels, mels_pred, mels_pred_postnet)
                m1_mi = self.mi_m1_net(lF1_embs, z)
                m1_mi_loss = self.weight_m1_mi * m1_mi
                loss = rec_loss + m1_mi_loss

            average_rec_loss += (rec_loss.item() - average_rec_loss) / batch_idx
            average_lld_m1_loss += (lld_m1_loss.item() - average_lld_m1_loss) / batch_idx
            average_m1_mi_loss += (m1_mi_loss.item() - average_m1_mi_loss) / batch_idx
            average_m1_mi += (m1_mi.item() - average_m1_mi) / batch_idx

        ctime = time.time()
        self.logger.info(f"Eval valid_set Epoch: {epoch} rec_loss: {average_rec_loss:.6f} "
                        f"lld_m1_loss: {average_lld_m1_loss:.6f} m1_mi_loss: {average_m1_mi_loss:.6f} used_time: {ctime-stime:.3f}s")
        to_train(all_models)
        
    def _train_epoch(self, epoch):
        '''
            Training logic for an epoch

            :param epoch: Integer, current training epoch.
            :return A log that contains average loss and metric in this epoch.
        '''
        average_rec_loss = average_lld_m1_loss = average_lld_m2_loss = average_m1_mi_loss = average_m2_mi_loss = 0
        stime = time.time()
        for batch_idx, (mels, lF1s, lF2s) in enumerate(self.data_loader, 1):
            mels, lF1s, lF2s = mels.float().to(self.device), lF1s.float().to(self.device), lF2s.float().to(self.device)

            # mi_first
            for j in range(5):
                lld_m1_loss, lld_m2_loss = self.mi_first(mels, lF1s, lF2s)

            # mi_second
            self.optimizer.zero_grad()
            z = self.encoder(mels)
            lF1_embs = self.encoder_F1(lF1s)
            lF2_embs = self.encoder_F2(lF2s)
            
            mels_pred, mels_pred_postnet = self.decoder(z, lF1_embs, lF2_embs)
            rec_loss = self.decoder.loss_function(mels, mels_pred, mels_pred_postnet)
            m1_mi = self.mi_m1_net(lF1_embs, z)
            m1_mi_loss = self.weight_m1_mi * m1_mi
            m2_mi = self.mi_m2_net(lF2_embs, z)
            m2_mi_loss = self.weight_m1_mi * m2_mi

            loss = rec_loss + m1_mi_loss + m2_mi_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            average_rec_loss += (rec_loss.item() - average_rec_loss) / batch_idx
            average_lld_m1_loss += (lld_m1_loss.item() - average_lld_m1_loss) / batch_idx
            average_lld_m2_loss += (lld_m2_loss.item() - average_lld_m2_loss) / batch_idx
            average_m1_mi_loss += (m1_mi_loss.item() - average_m1_mi_loss) / batch_idx
            average_m2_mi_loss += (m2_mi_loss.item() - average_m2_mi_loss) / batch_idx

            ctime = time.time()
            print(f"Train Epoch: {epoch} global_step: {self.global_step} rec_loss: {average_rec_loss:.6f} " 
                  f"lld_m1_loss: {average_lld_m1_loss:.6f} m1_mi_loss: {average_m1_mi_loss:.6f} "
                  f"lld_m2_loss: {average_lld_m2_loss:.6f} m2_mi_loss: {average_m2_mi_loss:.6f} "
                  f"used_time: {ctime-stime:.3f}s")
            self.global_step += 1
            stime = time.time()
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()
        self.logger.info(f"lr: {lr}")
        self.logger.info(f"Train Epoch: {epoch} global_step: {self.global_step} rec_loss: {average_rec_loss:.6f} "
                             f"lld_m1_loss: {average_lld_m1_loss:.6f} m1_mi_loss: {average_m1_mi_loss:.6f}"
                             f"lld_m2_loss: {average_lld_m2_loss:.6f} m2_mi_loss: {average_m2_mi_loss:.6f}")
