import os
import wandb
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from data_utils import decode_sequences

import lightning as L
from lightning.pytorch.callbacks import Callback

from models.Discriminator import Discriminator as D
from models.Generator import Generator as G

class AMPGANv3(L.LightningModule):
    def __init__(self, latent_dim , amp_dataset, generator_cfg, discriminator_cfg, generator_opt_cfg, discriminator_opt_cfg, run_num):
        super().__init__()
        self.latent_dim = latent_dim
        self.amp_dataset = amp_dataset
        self.max_length = self.amp_dataset.max_length
        self.n_tokens = len(self.amp_dataset.tokens)
        self.run_num = run_num

        os.system(f"rm -r results/{self.run_num}.txt")

        self.generator = self.build_generator(generator_cfg)
        self.discriminator_gan = self.build_discriminator(discriminator_cfg[0])
        self.discriminator_mic = self.build_discriminator(discriminator_cfg[1])
        
        self.generator_opt_cfg = generator_opt_cfg
        self.discriminator_opt_cfg = discriminator_opt_cfg

        self.eos_token = amp_dataset.tokens_dict['<EOS>']
        self.automatic_optimization = False

        self.fake_samples = None
    def build_generator(self, cfg):
        return G(output_shape=(self.max_length, self.n_tokens),
                 latent_shape=(self.latent_dim,),
                 species_shape=(len(self.amp_dataset.species),),
                 embed_dim=cfg.embed_dim)
    def build_discriminator(self, cfg):
        return D(model_type=cfg.names,
                 classes=cfg.classes,
                 n_tokens=self.n_tokens,
                 seq_len=self.max_length,
                 nhead=cfg.nhead,
                 nlayers=cfg.nlayers,
                 d_model=cfg.d_model,
                 d_hid=cfg.d_hid,
                 n_species=len(self.amp_dataset.species),
                 n_conditions=cfg.n_conditions,
                 dropout=cfg.dropout)
    
    def compute_loss(self, losses, length_scale=0.5, similarity_scale=0.5, gan_d_scale=2, mic_d_scale=1.0):
        return (length_scale * losses['length_loss']) + (similarity_scale * losses['similarity_loss']) + (gan_d_scale * losses['d_gan_loss']) + (mic_d_scale * losses['d_mic_loss'])
        # return (similarity_scale * losses['similarity_loss']) + (gan_d_scale * losses['d_gan_loss']) + (mic_d_scale * losses['d_mic_loss'])
    def gt_alive_mask(self, eos_pos_per_sample, L, buffer=1):
        """
        Hard mask covering positions 0 through eos_pos + buffer inclusive.
        buffer=1 includes the cterm position after EOS.
        """
        BS = eos_pos_per_sample.shape[0]
        position = torch.arange(L, device=self.device).unsqueeze(0).expand(BS, -1)

        return (position <= (eos_pos_per_sample + buffer).unsqueeze(1)).float()

    def freeze_discriminators(self):
        for d in [self.discriminator_gan, self.discriminator_mic]:
            d.eval() 
            for param in d.parameters():
                param.requires_grad = False

    def unfreeze_discriminators(self):
        for d in [self.discriminator_gan, self.discriminator_mic]:
            d.train() 
            for param in d.parameters():
                param.requires_grad = True    

    def length_extract (self, samples):
        eos_indices = (samples[:, self.eos_token, :] == 1).nonzero(as_tuple=False)
        eos_list = []
        for ind, eos in eos_indices:
            # <SOS>, <EOS>, nterminus, cterminus tokens are not included in the lengths
            # +1 is added because 'eos' is index 
            eos_list.append(eos - 3 + 1)

        # self.max_length - (4 special tokens) = max length of AA sequences 
        lengths_normalized = torch.tensor(eos_list).float()  / (self.max_length - 4)
        lengths_normalized = lengths_normalized.unsqueeze(1)

        return eos_indices, lengths_normalized

    def length_loss_target(self, samples, eos_pos):
        """Cross-entropy at GT EOS position - teaches WHERE to put EOS."""
        BS = samples.shape[0]
        eos_logits = samples[torch.arange(BS), eos_pos[:, 1]]
        targets = torch.full((BS,), self.eos_token, dtype=torch.long, device=self.device)
        return F.cross_entropy(eos_logits, targets)

    def length_loss_pre(self, samples, eos_pos):
        """Penalize EOS probability before GT - teaches NOT to EOS early."""
        BS, L, _ = samples.shape
        eos_pos_idx = eos_pos[:, 1]
        p_eos = F.softmax(samples, dim=2)[:, :, self.eos_token]
        position = torch.arange(L, device=self.device).unsqueeze(0).expand(BS, -1)
        pre_eos_mask = (position < eos_pos_idx.unsqueeze(1)).float()
        return (p_eos * pre_eos_mask).sum() / pre_eos_mask.sum().clamp(min=1)

    def length_loss(self, samples, eos_pos):
        loss = self.length_loss_target(samples, eos_pos)
        # loss = self.length_loss_target(samples, eos_pos) + self.length_loss_pre(samples, eos_pos)
        return loss

    def shuffle_samples(self, samples, labels):
        perm = torch.randperm(samples.size(0))
        samples = samples[perm]
        labels = labels[perm]
        return samples.float(), labels.long()
    
    def sample_sequences(self, samples):
        decoded_samples = decode_sequences(self.amp_dataset.tokens_dict, samples.permute(0,2,1).detach().cpu().numpy())
        arr = np.array(decoded_samples)
        return arr[np.random.choice(arr.shape[0], 2, replace=False)]

    def discriminator_step(self, batch, discriminator_type):
        BS = len(batch['samples'])

        samples = batch['samples']
        conditions = batch['conditions']

        species_n = conditions[:, 0:6].int()
        mic_value = conditions[:, 16].float()

        eos_indices, lengths = self.length_extract(samples)
        
        d_loss = 0  
        
        # when discriminator type is GAN
        if discriminator_type == "GAN":
            self.generator.eval()
            with torch.no_grad():
                fake_samples = self.generator(
                    torch.randn(BS, self.latent_dim).to(self.device), 
                    species_n.to(self.device),
                    mic_value.to(self.device), 
                    lengths.to(self.device)
                )
            self.generator.train()

            fake_samples = F.gumbel_softmax(fake_samples, tau=1.0, hard=True, dim=2)

            # Hard masks
            L = samples.shape[2]
            real_mask = self.gt_alive_mask(eos_indices[:, 1], L, buffer=1)  # [BS, L]
            
            # For fakes, find predicted EOS position (first occurrence per row)
            fake_tokens = fake_samples.argmax(dim=2)  # [BS, L]
            fake_eos_hits = (fake_tokens == self.eos_token)  # [BS, L]
            # First EOS position per sample; if no EOS, use L-1
            has_eos = fake_eos_hits.any(dim=1)
            fake_eos_pos = torch.where(
                has_eos,
                fake_eos_hits.float().argmax(dim=1),  # argmax returns first True
                torch.full((BS,), L - 1, device=self.device, dtype=torch.long)
            )
            fake_mask = self.gt_alive_mask(fake_eos_pos, L, buffer=1)  # [BS, L]
            
            # Apply masks: zero out post-EOS+cterm positions
            # samples is [BS, n_tokens, L], fake_samples is [BS, L, n_tokens]
            masked_real = samples * real_mask.unsqueeze(1)  # broadcast over n_tokens
            masked_fake = (fake_samples * fake_mask.unsqueeze(2)).permute(0, 2, 1)  # [BS, n_tokens, L]
            
            combined_samples = torch.cat([masked_real, masked_fake], dim=0)
            labels = torch.cat([torch.ones(BS), torch.zeros(BS)], dim=0)
            combined_samples, labels = self.shuffle_samples(combined_samples, labels)
            output = self.discriminator_gan(combined_samples.to(self.device))
            d_loss = F.cross_entropy(output, labels.to(self.device))
            self.fake_samples = fake_samples.clone()
        
        elif discriminator_type == "MIC":
            L = samples.shape[2]
            real_mask = self.gt_alive_mask(eos_indices[:, 1], L, buffer=1)
            masked_real = samples * real_mask.unsqueeze(1)
            
            output = self.discriminator_mic(masked_real.float().to(self.device), species_n.to(self.device))
            d_loss = F.mse_loss(output, mic_value.to(self.device))
        return d_loss


    def generator_step(self, batch):

        BS = len(batch['samples'])

        samples = batch['samples']
        conditions = batch['conditions']

        species_n = conditions[:, 0:6].int()
        mic_value = conditions[:, 16].float()

        eos_indices, lengths = self.length_extract(samples)
        
        # generate samples
        fake_samples = self.generator(
            torch.randn(BS, self.latent_dim).to(self.device), 
            species_n.to(self.device),
            mic_value.to(self.device), 
            lengths.to(self.device)
        )

        # legnths and similarities loss
        
        length_loss = self.length_loss(fake_samples, eos_indices)
        # Build a mask that's 1 up to and including GT EOS, 0 after

        BS, L = samples.shape[0], samples.shape[2]
        eos_pos_per_sample = eos_indices[:, 1] + 1  # assumes one EOS per real, which is true
        position = torch.arange(L, device=self.device).unsqueeze(0).expand(BS, -1)
        valid_mask = position <= eos_pos_per_sample.unsqueeze(1)  # [BS, L]

        # Per-position CE, then mask
        ce_per_pos = F.cross_entropy(
            fake_samples.permute(0, 2, 1),
            samples.argmax(dim=1),
            reduction='none'
        )  # [BS, L]

        # Pre-EOS positions get full weight
        pre_and_eos_weight = valid_mask.float()  # 1 for pos <= GT_EOS+1, else 0

        # Post-EOS positions get small weight (e.g., 0.1)
        post_eos_weight = (~valid_mask).float() * 0.1

        weights = pre_and_eos_weight + post_eos_weight
        similarity_loss = (ce_per_pos * weights).sum() / weights.sum()
        # similarity_loss = F.cross_entropy(fake_samples.to(self.device), samples.permute(0,2,1).to(self.device))


        gum_fake_samples = F.gumbel_softmax(fake_samples, tau=1.0, hard=True, dim=2)
        fake_samples = F.softmax(fake_samples, dim=2)
        self.freeze_discriminators()

        # D1
        d_gan_output = self.discriminator_gan(gum_fake_samples.permute(0,2,1).to(self.device))
        d_gan_loss = F.cross_entropy(d_gan_output.to(self.device), torch.ones(BS,dtype=torch.long, device=self.device))

        # D2

        d_mic_output = self.discriminator_mic(fake_samples.permute(0,2,1).to(self.device), species_n.int().to(self.device))
        d_mic_loss = F.mse_loss(d_mic_output.to(self.device), mic_value.to(self.device))

        self.unfreeze_discriminators()

        losses = {'length_loss':length_loss, 'similarity_loss':similarity_loss, 'd_gan_loss':d_gan_loss, 'd_mic_loss':d_mic_loss}

        # losses = {'similarity_loss':similarity_loss, 'd_gan_loss':d_gan_loss, 'd_mic_loss':d_mic_loss}
        
        self.log('length_loss', length_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('similarity_loss', similarity_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('d_gan_loss', d_gan_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('d_mic_loss', d_mic_loss, on_step=False, on_epoch=True, prog_bar=True)

        return self.compute_loss(losses)
    def training_step(self, batch, batch_idx):
        opt_d1, opt_d2, opt_g = self.optimizers()
        scheduler = self.lr_schedulers()

        # ------------------
        # Discriminator 1 (GAN)
        # ------------------
        loss_d1 = self.discriminator_step(batch, discriminator_type="GAN")
        opt_d1.zero_grad()
        self.manual_backward(loss_d1)
        opt_d1.step()
        self.log("train/d1_loss", loss_d1, on_step=False, on_epoch=True, prog_bar=True)

        # ------------------
        # Discriminator 2 (MIC)
        # ------------------
        loss_d2 = self.discriminator_step(batch, discriminator_type="MIC")
        opt_d2.zero_grad()
        self.manual_backward(loss_d2)
        opt_d2.step()
        self.log("train/d2_loss", loss_d2, on_step=False, on_epoch=True, prog_bar=True)

        # ------------------
        # Generator
        # ------------------
        loss_g = self.generator_step(batch)
        opt_g.zero_grad()
        self.manual_backward(loss_g)
        clip_grad_norm_(self.generator.parameters(), max_norm=3.0)
        opt_g.step()

        self.log("train/g_loss", loss_g, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss_d1": loss_d1, "loss_d2": loss_d2, "loss_g": loss_g}
    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()
        if self.current_epoch % 10 == 0:
            random_samples = self.sample_sequences(self.fake_samples)
            with open(f"results/{self.run_num}.txt", "a") as f:
                f.write(f"{self.current_epoch}:     {random_samples[0]}   {random_samples[1]}\n")

        
    def configure_optimizers(self):
        optimizer_seq = optim.AdamW(self.discriminator_gan.parameters(), lr=self.discriminator_opt_cfg[0].lr, betas=self.discriminator_opt_cfg[0].betas, weight_decay=self.discriminator_opt_cfg[0].weight_decay)
        optimizer_mic = optim.AdamW(self.discriminator_mic.parameters(), lr=self.discriminator_opt_cfg[1].lr, betas=self.discriminator_opt_cfg[1].betas)
        optimizer_g = optim.AdamW(self.generator.parameters(),  lr=self.generator_opt_cfg.lr, betas=self.generator_opt_cfg.betas)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=50, gamma=0.5)

        return [optimizer_seq, optimizer_mic, optimizer_g], [scheduler]


class GANCheckpoint(Callback):
    def __init__(self, run_num, every_n_epochs=50, save_dir="logs/", ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.save_dir = save_dir
        self.run_num = run_num
        os.makedirs(os.path.join(save_dir, "Generator"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Discriminator1"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "Discriminator2"), exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0 and epoch != 0:
            torch.save(pl_module.generator.state_dict(),
                       f"{self.save_dir}/Generator/Generator_{self.run_num}_{epoch}.pth")
            torch.save(pl_module.discriminator_gan.state_dict(),
                       f"{self.save_dir}/Discriminator1/Discriminator1_{self.run_num}_{epoch}.pth")
            torch.save(pl_module.discriminator_mic.state_dict(),
                       f"{self.save_dir}/Discriminator2/Discriminator2_{self.run_num}_{epoch}.pth")
