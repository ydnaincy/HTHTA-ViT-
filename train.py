#!/usr/bin/env python3
"""
HTHTA-ViT++: An Explainable and Efficient Vision Transformer
with Hierarchical GRU-Guided Token Attention

Training script for HTHTA-ViT++ model on various datasets.
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiGRU(nn.Module):
    """Bidirectional GRU module for token sequencing."""
    
    def __init__(self, input_dim=768, hidden_dim=768, num_layers=2):
        super(BiGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.bigru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True
        )
        self.projection = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass through BiGRU
        out, _ = self.bigru(x, h0)
        
        # Project back to original dimension
        out = self.projection(out)
        
        return out


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling with interpretability."""
    
    def __init__(self, input_dim=768, num_heads=8, head_dim=96):
        super(MultiHeadAttentionPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, head_dim),
                nn.Tanh(),
                nn.Linear(head_dim, 1)
            ) for _ in range(num_heads)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape
        
        head_outputs = []
        attention_weights = []
        
        for head in self.attention_heads:
            # Compute attention scores
            scores = head(x)  # (batch_size, seq_len, 1)
            weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch_size, seq_len)
            
            # Apply attention weights
            context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)
            
            head_outputs.append(context)
            attention_weights.append(weights)
        
        # Concatenate all head outputs
        pooled_output = torch.cat(head_outputs, dim=1)  # (batch_size, num_heads * input_dim)
        
        return pooled_output, attention_weights


class HierarchicalCLSFusion(nn.Module):
    """Hierarchical CLS-token fusion mechanism."""
    
    def __init__(self, input_dim=768, num_heads=8):
        super(HierarchicalCLSFusion, self).__init__()
        self.projection = nn.Linear(num_heads * input_dim, input_dim)
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, cls_token, pooled_token):
        # Project pooled token to match CLS token dimension
        pooled_projected = self.projection(pooled_token)
        
        # Hierarchical fusion
        interaction = cls_token * pooled_projected
        fused = self.gamma * cls_token + (1 - self.gamma) * pooled_projected + self.beta * interaction
        
        return fused


class HTHTAViTPlusPlus(nn.Module):
    """HTHTA-ViT++ model implementation."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super(HTHTAViTPlusPlus, self).__init__()
        
        # Load pretrained ViT backbone
        self.vit = torchvision.models.vit_b_16(pretrained=pretrained)
        
        # Remove the classification head
        self.vit.heads = nn.Identity()
        
        # BiGRU module
        self.bigru = BiGRU(input_dim=768, hidden_dim=768, num_layers=2)
        
        # Multi-head attention pooling
        self.attention_pooling = MultiHeadAttentionPooling(input_dim=768, num_heads=8)
        
        # Hierarchical CLS fusion
        self.cls_fusion = HierarchicalCLSFusion(input_dim=768, num_heads=8)
        
        # Classification head
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        # Extract features from ViT backbone
        # We need to modify this to get intermediate features
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.vit._process_input(x)
        n = x.shape[0]
        
        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Add positional encoding
        x = self.vit.encoder.pos_embedding + x
        x = self.vit.encoder.dropout(x)
        
        # Pass through transformer layers
        for layer in self.vit.encoder.layers:
            x = layer(x)
        
        # Apply layer norm
        x = self.vit.encoder.ln(x)
        
        # Separate CLS token and patch tokens
        cls_token = x[:, 0]  # (batch_size, 768)
        patch_tokens = x[:, 1:]  # (batch_size, seq_len, 768)
        
        # Apply BiGRU to patch tokens
        bigru_output = self.bigru(patch_tokens)
        
        # Apply attention pooling
        pooled_output, attention_weights = self.attention_pooling(bigru_output)
        
        # Hierarchical CLS fusion
        fused_features = self.cls_fusion(cls_token, pooled_output)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights


def get_transforms(dataset_name, train=True):
    """Get data transforms for different datasets."""
    
    if dataset_name.lower() in ['cifar10', 'cifar100']:
        if train:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        # For other datasets (Intel, Tiny-ImageNet)
        if train:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    return transform


def get_dataset(dataset_name, data_dir, train=True):
    """Load dataset based on name."""
    
    transform = get_transforms(dataset_name, train)
    
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=train, download=True, transform=transform
        )
    else:
        # For custom datasets like Intel or Tiny-ImageNet
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(data_dir, 'train' if train else 'test'),
            transform=transform
        )
    
    return dataset


def calculate_fap(attention_weights, gradcam_maps, threshold=0.15):
    """Calculate Focused Attention Percentage (FAP)."""
    fap_scores = []
    
    for attn, gradcam in zip(attention_weights, gradcam_maps):
        # Convert attention weights to spatial map (this is simplified)
        # In practice, you'd need to reshape based on patch positions
        
        # Get top threshold% attention regions
        top_k = int(threshold * len(attn))
        _, top_indices = torch.topk(attn, top_k)
        
        # Calculate overlap with ground truth (simplified)
        # This is a placeholder - actual implementation depends on your GradCAM setup
        overlap = len(set(top_indices.cpu().numpy()) & set(gradcam)) / len(gradcam)
        fap_scores.append(overlap)
    
    return np.mean(fap_scores)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits, attention_weights = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    return total_loss / len(train_loader), accuracy


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            logits, attention_weights = model(data)
            loss = criterion(logits, target)
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    return total_loss / len(val_loader), accuracy, all_preds, all_targets


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_classification_report(y_true, y_pred, class_names, save_path):
    """Save classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as text
    text_report = classification_report(y_true, y_pred, target_names=class_names)
    with open(save_path.replace('.json', '.txt'), 'w') as f:
        f.write(text_report)


def main():
    parser = argparse.ArgumentParser(description='Train HTHTA-ViT++ model')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100', 'intel', 'tiny_imagenet'],
                        help='Dataset to train on')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ViT backbone')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f'Using device: {device}')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'{args.dataset}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'tensorboard')
    
    # Load datasets
    train_dataset = get_dataset(args.dataset, args.data_dir, train=True)
    val_dataset = get_dataset(args.dataset, args.data_dir, train=False)
    
    # Get number of classes
    if args.dataset == 'cifar10':
        num_classes = 10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'cifar100':
        num_classes = 100
        class_names = [f'class_{i}' for i in range(100)]
    elif args.dataset == 'intel':
        num_classes = 6
        class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
        class_names = [f'class_{i}' for i in range(200)]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create model
    model = HTHTAViTPlusPlus(num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_acc = 0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        logger.info(f'Resumed from epoch {start_epoch} with best accuracy {best_acc:.2f}%')
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }
        
        torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')
        
        if is_best:
            torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
            
            # Save confusion matrix and classification report for best model
            save_confusion_matrix(
                val_targets, val_preds, class_names,
                output_dir / 'confusion_matrix_best.png'
            )
            save_classification_report(
                val_targets, val_preds, class_names,
                output_dir / 'classification_report_best.json'
            )
        
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.info(f'Best Acc: {best_acc:.2f}%')
        logger.info('-' * 50)
    
    # Final evaluation
    logger.info('Training completed!')
    logger.info(f'Best validation accuracy: {best_acc:.2f}%')
    
    # Save final results
    results = {
        'dataset': args.dataset,
        'best_accuracy': best_acc,
        'total_epochs': args.epochs,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'final_train_accuracy': train_acc,
        'final_val_accuracy': val_acc,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    writer.close()


if __name__ == '__main__':
    main()
