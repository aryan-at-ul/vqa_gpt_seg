import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from src.training_wrapper import VQGPTSegmentation
from src.utils.data_loading import ISICDataset, get_transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--test-data-dir', type=str, default='/home/annatar/projects/datasets/ISIC2016/test')
    parser.add_argument('--test-mask-dir', type=str, default='/home/annatar/projects/datasets/ISIC2016/test_masks')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--output-dir', type=str, default='test_results')
    return parser.parse_args()

def save_predictions(images, masks, preds, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img, mask, pred, fname) in enumerate(zip(images, masks, preds, filenames)):
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(mask.cpu().numpy(), cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(pred.cpu().numpy(), cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{fname}_comparison.png'))
        plt.close()

def main():
    args = parse_args()
    
    # Create test dataset and loader
    _, test_transform = get_transforms(args.image_size)
    test_dataset = ISICDataset(args.test_data_dir, args.test_mask_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    model = VQGPTSegmentation.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    model.cuda()
    
    # Initialize metrics
    all_preds = []
    all_masks = []
    
    # Test loop
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            masks = masks.cuda()
            
            # Forward pass
            logits = model(images)
            preds = torch.sigmoid(logits)
            
            # Store predictions and masks for metrics
            all_preds.extend(preds[:,1].cpu().numpy().ravel())  # Take class 1 probabilities
            all_masks.extend(masks.cpu().numpy().ravel())
            
            # Save some prediction visualizations
            if i < 10:  # Save first 10 batches
                pred_masks = (preds[:,1] > 0.5).float()
                save_predictions(
                    images, masks, pred_masks,
                    [f'batch_{i}_sample_{j}' for j in range(len(images))],
                    os.path.join(args.output_dir, 'visualizations')
                )
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)
    
    # Calculate PR and ROC curves
    precision, recall, _ = precision_recall_curve(all_masks, all_preds)
    fpr, tpr, _ = roc_curve(all_masks, all_preds)
    
    # Calculate AUC scores
    pr_auc = auc(recall, precision)
    roc_auc = auc(fpr, tpr)
    
    # Plot and save curves
    plt.figure(figsize=(10, 5))
    
    # PR curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics.png'))
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f'PR AUC: {pr_auc:.3f}\n')
        f.write(f'ROC AUC: {roc_auc:.3f}\n')

if __name__ == '__main__':
    main()