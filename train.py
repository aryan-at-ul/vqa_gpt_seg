import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.training_wrapper import VQGPTSegmentation
from src.utils.data_loading import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--train-data-dir', type=str, default='/home/annatar/projects/datasets/ISIC2016/train')
    parser.add_argument('--train-mask-dir', type=str, default='/home/annatar/projects/datasets/ISIC2016/train_masks')
    parser.add_argument('--val-data-dir', type=str, default='/home/annatar/projects/datasets/ISIC2016/test')
    parser.add_argument('--val-mask-dir', type=str, default='/home/annatar/projects/datasets/ISIC2016/test_masks')
    parser.add_argument('--image-size', type=int, default=224)
    

    parser.add_argument('--batch-size', type=int, default=1) # bad result vqa accumulation of gradients
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    

    parser.add_argument('--vq-embed-dim', type=int, default=256)
    parser.add_argument('--vq-n-embed', type=int, default=1024)
    parser.add_argument('--vq-hidden-channels', type=int, default=128)
    parser.add_argument('--vq-n-res-blocks', type=int, default=2)
    parser.add_argument('--disc-start', type=int, default=10000)
    parser.add_argument('--disc-weight', type=float, default=0.8)
    parser.add_argument('--perceptual-weight', type=float, default=1.0)
    parser.add_argument('--codebook-weight', type=float, default=1.0)
    
    # no changes as it is >> https://github.com/openai/image-gpt
    parser.add_argument('--gpt-n-layer', type=int, default=8)
    parser.add_argument('--gpt-n-head', type=int, default=4)
    parser.add_argument('--gpt-n-embd', type=int, default=256)
    

    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--experiment-name', type=str, default='vqgpt_segmentation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    pl.seed_everything(42)
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    vqgan_config = {
        'n_embed': args.vq_n_embed,
        'embed_dim': args.vq_embed_dim,
        'hidden_channels': args.vq_hidden_channels,
        'n_res_blocks': args.vq_n_res_blocks,
        'disc_start': args.disc_start,
        'disc_weight': args.disc_weight,
        'perceptual_weight': args.perceptual_weight,
        'codebook_weight': args.codebook_weight
    }
    

    gpt_config = {
        'vocab_size': args.vq_n_embed,
        'block_size': (args.image_size // 4) ** 2,
        'n_layer': args.gpt_n_layer,
        'n_head': args.gpt_n_head,
        'n_embd': args.gpt_n_embd
    }
    

    segmentation_config = {
        'input_dim': args.gpt_n_embd,
        'num_classes': 2,  # binary segmentation, to test on higher classes
        'hidden_dim': args.gpt_n_embd // 2
    }
    

    if args.checkpoint_path:
        model = VQGPTSegmentation.load_from_checkpoint(
            args.checkpoint_path,
            vqgan_config=vqgan_config,
            gpt_config=gpt_config,
            segmentation_config=segmentation_config,
            learning_rate=args.learning_rate
        )
    else:
        model = VQGPTSegmentation(
            vqgan_config=vqgan_config,
            gpt_config=gpt_config,
            segmentation_config=segmentation_config,
            learning_rate=args.learning_rate
        )
    

    train_loader, val_loader = create_dataloaders(
        train_data_dir=args.train_data_dir,
        train_mask_dir=args.train_mask_dir,
        val_data_dir=args.val_data_dir,
        val_mask_dir=args.val_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename=f'{args.experiment_name}' + '-{epoch:02d}-{val_seg_iou:.2f}',
        monitor='val/seg/iou',
        mode='max',
        save_top_k=3,
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    

    wandb_logger = WandbLogger(
        project=args.experiment_name,
        name=f"{args.experiment_name}-{args.image_size}-bs{args.batch_size}-lr{args.learning_rate}"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger,
        log_every_n_steps=10,
        strategy='auto'
    )
    

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
