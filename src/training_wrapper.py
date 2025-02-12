# src/training_wrapper.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex
from contextlib import contextmanager
from src.vqa.vqagan import VQGAN
from src.imagegpt_model import ImageGPT
from src.segmentation_head import SegmentationHead
import wandb

class VQGPTSegmentation(pl.LightningModule):
    def __init__(self, vqgan_config, gpt_config, segmentation_config, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.vqgan = VQGAN(**vqgan_config)
        self.gpt = ImageGPT(**gpt_config)
        self.segmentation_head = SegmentationHead(**segmentation_config)
        

        self.train_accuracy = Accuracy(task="multiclass", num_classes=segmentation_config['num_classes'])
        self.val_accuracy = Accuracy(task="multiclass", num_classes=segmentation_config['num_classes'])
        self.val_iou = JaccardIndex(task="multiclass", num_classes=segmentation_config['num_classes'])
        
        self.learning_rate = learning_rate
        self.num_classes = segmentation_config['num_classes']
        self.automatic_optimization = False  


        for name, param in self.vqgan.discriminator.named_parameters():
            print(f"Discriminator parameter '{name}' requires_grad: {param.requires_grad}")

    @contextmanager
    def ema_scope(self):
        if hasattr(self.vqgan.quantize, "embedding_ema"):
            self.vqgan.quantize.embedding_ema = True
        try:
            yield
        finally:
            if hasattr(self.vqgan.quantize, "embedding_ema"):
                self.vqgan.quantize.embedding_ema = False

    def forward(self, x):
        
        pass

    def training_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)

    
        opt_ae, opt_disc, opt_gpt = self.optimizers()


        images = 2.0 * images - 1.0

     
        opt_ae.zero_grad()
        loss_vq, vq_logs = self.vqgan(images, optimizer_idx=0, global_step=self.global_step)  # Unpack two values

  
        loss_vq = loss_vq.mean() if loss_vq.dim() > 0 else loss_vq

        self.manual_backward(loss_vq, retain_graph=True)
        opt_ae.step()

    
        self.log("train/vqgan/loss_vq", loss_vq, prog_bar=True, on_step=True, on_epoch=True)

 
        loss_disc = None
        disc_logs = None
        if self.global_step >= self.vqgan.loss.disc_start:
            opt_disc.zero_grad()
            loss_disc, disc_logs = self.vqgan(images, optimizer_idx=1, global_step=self.global_step)  # Unpack two values

            if loss_disc is not None:
                loss_disc = loss_disc.mean() if loss_disc.dim() > 0 else loss_disc
                self.manual_backward(loss_disc)
                opt_disc.step()

         
                self.log("train/vqgan/loss_disc", loss_disc, prog_bar=True, on_step=True, on_epoch=True)
            else:
                self.log("train/vqgan/loss_disc", 0.0, prog_bar=True, on_step=True, on_epoch=True)
        else:
     
            self.log("train/vqgan/loss_disc", 0.0, prog_bar=True, on_step=True, on_epoch=True)
            # print("Skipping backward and step for discriminator loss (disc_start not reached).")

        #### Train GPT and Segmentation head ####
        opt_gpt.zero_grad()
        with self.ema_scope():
          
            quant, emb_loss, info = self.vqgan.encode(images)  # Unpack three values
            # print(f"Type of indices in training_step: {type(info)}")  # Should output: <class 'torch.Tensor'> or <class 'tuple'>

   
            if isinstance(info, tuple):
                # this is original, from tamming transformers, not used in the simple vqa
                if len(info) >= 3:
                    _, _, indices = info
                else:
                    raise ValueError(f"Info tuple has insufficient elements: {len(info)}")
            elif isinstance(info, torch.Tensor):
                indices = info
            else:
                raise TypeError(f"Expected 'info' to be a tuple or tensor, but got {type(info)}")

  
            batch_size = images.shape[0]
            h = w = images.shape[2] // 4  # VQGAN's downsampling factor is 4
            try:
                indices = indices.reshape(batch_size, h * w)
            except Exception as e:
                raise ValueError(f"Error reshaping indices: {e}")


            features = self.gpt(indices)


            segmentation = self.segmentation_head(features)

  
            seg_loss = F.cross_entropy(segmentation, masks)

 
        seg_loss = seg_loss.mean() if seg_loss.dim() > 0 else seg_loss

        self.manual_backward(seg_loss)
        opt_gpt.step()


        with torch.no_grad():
            pred_masks = torch.argmax(segmentation, dim=1)
            accuracy = self.train_accuracy(pred_masks, masks)

        self.log_dict({
            'train/seg/loss': seg_loss,
            'train/seg/accuracy': accuracy,
        }, prog_bar=True, on_step=True, on_epoch=True)

    
        if vq_logs is not None:
            self.log_dict({
                'train/vqgan/total_loss': vq_logs.get("total_loss", 0.0),
                'train/vqgan/rec_loss': vq_logs.get("rec_loss", 0.0),
                'train/vqgan/codebook_loss': vq_logs.get("codebook_loss", 0.0),
                'train/vqgan/g_loss': vq_logs.get("g_loss", 0.0),
            }, prog_bar=True, on_step=True, on_epoch=True)

        if loss_disc is not None and disc_logs is not None:
            self.log_dict({
                'train/vqgan/disc_loss_real': disc_logs.get("disc_loss_real", 0.0),
                'train/vqgan/disc_loss_fake': disc_logs.get("disc_loss_fake", 0.0),
            }, prog_bar=True, on_step=True, on_epoch=True)


        del loss_vq, loss_disc, seg_loss, vq_logs, disc_logs, features, segmentation, pred_masks, accuracy
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        images = images.to(self.device)
        masks = masks.to(self.device)
        images = 2.0 * images - 1.0  

        with self.ema_scope():
         
            reconstructions, codebook_loss, info = self.vqgan(images)  # Unpack three values
            # print(f"Type of indices in validation_step: {type(info)}")  # Should output: <class 'torch.Tensor'> or <class 'tuple'>

           
            if isinstance(info, tuple):
              
                if len(info) >= 3:
                    _, _, indices = info
                else:
                    raise ValueError(f"Info tuple has insufficient elements: {len(info)}")
            elif isinstance(info, torch.Tensor):
                indices = info
            else:
                raise TypeError(f"Expected 'info' to be a tuple or tensor, but got {type(info)}")


            batch_size = images.shape[0]
            h = w = images.shape[2] // 4
            try:
                indices = indices.reshape(batch_size, h * w)
            except Exception as e:
                raise ValueError(f"Error reshaping indices: {e}")

            features = self.gpt(indices)
            segmentation = self.segmentation_head(features)

            seg_loss = F.cross_entropy(segmentation, masks)
            pred_masks = torch.argmax(segmentation, dim=1)
            accuracy = self.val_accuracy(pred_masks, masks)
            iou = self.val_iou(pred_masks, masks)


            rec_loss = torch.abs(images - reconstructions).mean()


            # print(f"Validation Step: seg_loss={seg_loss.item()}, accuracy={accuracy.item()}, iou={iou.item()}, rec_loss={rec_loss.item()}, codebook_loss={codebook_loss.item()}")

        self.log_dict({
            'val/seg/loss': seg_loss,
            'val/seg/accuracy': accuracy,
            'val/seg/iou': iou,
            'val/vqgan/rec_loss': rec_loss,
            'val/vqgan/codebook_loss': codebook_loss,
        }, prog_bar=True, on_step=False, on_epoch=True)


        if batch_idx == 0:
  
            self._log_images(
                images=images,
                reconstructions=reconstructions,
                masks=masks,
                pred_masks=pred_masks
            )

        return seg_loss

    def _log_images(self, images, reconstructions, masks, pred_masks):
        # for 1 only [-1, 1] to [0, 1]
        images = (images + 1.0) / 2.0
        reconstructions = (reconstructions + 1.0) / 2.0


        self.logger.experiment.log({
            "val/examples": [
                wandb.Image(img, caption=f"Sample {i}") for i, img in enumerate(images.cpu())
            ],
            "val/reconstructions": [
                wandb.Image(rec, caption=f"Reconstruction {i}") for i, rec in enumerate(reconstructions.cpu())
            ],
            "val/masks": [
                wandb.Image(mask.unsqueeze(0).float().cpu(), caption=f"Mask {i}") for i, mask in enumerate(masks)
            ],
            "val/predictions": [
                wandb.Image(pred.unsqueeze(0).float().cpu(), caption=f"Prediction {i}") for i, pred in enumerate(pred_masks)
            ],
            "global_step": self.global_step
        })

    def configure_optimizers(self):
     
        opt_ae = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.quantize.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=self.learning_rate * 0.1,  
            betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(
            self.vqgan.discriminator.parameters(),
            lr=self.learning_rate * 0.1,  
            betas=(0.5, 0.9)
        )
        

        gpt_seg_opt = torch.optim.AdamW(
            list(self.gpt.parameters()) + 
            list(self.segmentation_head.parameters()),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                gpt_seg_opt,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate / 10
            ),
            "interval": "epoch",
            "frequency": 1
        }
        
        return [opt_ae, opt_disc, gpt_seg_opt], [scheduler]
