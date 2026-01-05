"""
Training script for multimodal sequence modeling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import json
import os
from datetime import datetime
from pathlib import Path

from model import MultimodalStoryModel, ImprovedMultimodalStoryModel
from data_loader import StoryDataset
from evaluate import calculate_metrics
from utils import save_checkpoint, load_checkpoint, plot_training_curves


class Trainer:
    """Trainer class for multimodal sequence modeling"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss functions
        self.criterion = self._setup_loss_functions()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        
        # Create output directories
        self._create_directories()
        
        # Initialize wandb if enabled
        if config.get('logging', {}).get('wandb', False):
            wandb.init(
                project=config['project']['name'],
                config=config,
                name=f"{config['project']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        optimizer_type = self.config['training'].get('optimizer', 'adamw')
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _setup_loss_functions(self):
        """Setup loss functions for multi-task learning"""
        criterion = {}
        
        # Text generation loss (cross entropy)
        criterion['text'] = nn.CrossEntropyLoss(
            ignore_index=0,  # ignore padding
            label_smoothing=0.1  # label smoothing for better generalization
        )
        
        # Image reconstruction loss
        criterion['image'] = nn.MSELoss()
        
        # Optional: perceptual loss for images
        if self.config.get('innovation', {}).get('use_perceptual_loss', False):
            # You would need to implement a perceptual loss using VGG
            pass
        
        # Multi-task reconstruction loss
        criterion['reconstruction'] = nn.MSELoss()
        
        # Coherence loss (binary cross entropy)
        criterion['coherence'] = nn.BCELoss()
        
        return criterion
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6),
                verbose=True
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            return None
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['paths']['checkpoints'],
            self.config['paths']['results'],
            self.config['paths']['logs'],
            os.path.join(self.config['paths']['results'], 'baseline'),
            os.path.join(self.config['paths']['results'], 'improved'),
            os.path.join(self.config['paths']['results'], 'comparative_analysis'),
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader, curriculum=False):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_text_loss = 0
        total_image_loss = 0
        total_mtl_loss = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} - Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            images = batch['images'].to(self.device)
            text = batch['text'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            targets = batch['target_text'].to(self.device)
            target_images = batch['target_images'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, ImprovedMultimodalStoryModel):
                outputs = self.model(
                    images, text, text_lengths,
                    teacher_forcing=True,
                    targets=targets,
                    return_intermediate=True
                )
                
                # Calculate losses
                # Text generation loss
                text_pred = outputs['text_predictions']
                text_loss = self.criterion['text'](
                    text_pred.view(-1, text_pred.size(-1)),
                    targets.view(-1)
                )
                
                # Image reconstruction loss
                image_pred = outputs['image_predictions']
                image_loss = self.criterion['image'](image_pred, target_images)
                
                # Multi-task learning losses
                if 'reconstruction_output' in outputs:
                    reconstruction_loss = self.criterion['reconstruction'](
                        outputs['reconstruction_output'],
                        torch.cat([outputs['visual_features'], outputs['text_features']], dim=-1)
                    )
                else:
                    reconstruction_loss = 0
                
                if 'coherence_scores' in outputs:
                    # Create coherence labels (1 for coherent sequences)
                    coherence_labels = torch.ones_like(outputs['coherence_scores'])
                    coherence_loss = self.criterion['coherence'](
                        outputs['coherence_scores'],
                        coherence_labels
                    )
                else:
                    coherence_loss = 0
                
                # Total loss with weights
                mtl_weights = self.config['innovation']['multi_task']
                total_batch_loss = (
                    text_loss * 0.5 +
                    image_loss * 0.3 +
                    reconstruction_loss * mtl_weights.get('reconstruction_weight', 0.2) +
                    coherence_loss * mtl_weights.get('coherence_weight', 0.1)
                )
                
                # Store individual losses
                total_text_loss += text_loss.item()
                total_image_loss += image_loss.item()
                total_mtl_loss += (reconstruction_loss + coherence_loss).item()
                
            else:
                # Baseline model
                image_pred, text_pred = self.model(
                    images, text, text_lengths,
                    teacher_forcing=True,
                    targets=targets
                )
                
                # Calculate losses
                text_loss = self.criterion['text'](
                    text_pred.view(-1, text_pred.size(-1)),
                    targets.view(-1)
                )
                image_loss = self.criterion['image'](image_pred, target_images)
                
                total_batch_loss = text_loss * 0.5 + image_loss * 0.5
                total_text_loss += text_loss.item()
                total_image_loss += image_loss.item()
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update statistics
            total_loss += total_batch_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'text_loss': total_text_loss / (batch_idx + 1),
                'img_loss': total_image_loss / (batch_idx + 1)
            })
            
            # Log to wandb
            if self.config.get('logging', {}).get('wandb', False):
                wandb.log({
                    'train/batch_loss': total_batch_loss.item(),
                    'train/text_loss': text_loss.item() if 'text_loss' in locals() else 0,
                    'train/image_loss': image_loss.item() if 'image_loss' in locals() else 0,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_text_loss = 0
        total_image_loss = 0
        
        # Collect predictions and targets for metrics calculation
        all_text_predictions = []
        all_text_targets = []
        all_image_predictions = []
        all_image_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} - Validation")
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                images = batch['images'].to(self.device)
                text = batch['text'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                targets = batch['target_text'].to(self.device)
                target_images = batch['target_images'].to(self.device)
                
                # Forward pass
                if isinstance(self.model, ImprovedMultimodalStoryModel):
                    outputs = self.model(
                        images, text, text_lengths,
                        teacher_forcing=False,
                        return_intermediate=True
                    )
                    
                    # Calculate losses
                    text_pred = outputs['text_predictions']
                    text_loss = self.criterion['text'](
                        text_pred.view(-1, text_pred.size(-1)),
                        targets.view(-1)
                    )
                    
                    image_pred = outputs['image_predictions']
                    image_loss = self.criterion['image'](image_pred, target_images)
                    
                    # Multi-task learning losses
                    if 'reconstruction_output' in outputs:
                        reconstruction_loss = self.criterion['reconstruction'](
                            outputs['reconstruction_output'],
                            torch.cat([outputs['visual_features'], outputs['text_features']], dim=-1)
                        )
                    else:
                        reconstruction_loss = 0
                    
                    if 'coherence_scores' in outputs:
                        coherence_labels = torch.ones_like(outputs['coherence_scores'])
                        coherence_loss = self.criterion['coherence'](
                            outputs['coherence_scores'],
                            coherence_labels
                        )
                    else:
                        coherence_loss = 0
                    
                    mtl_weights = self.config['innovation']['multi_task']
                    batch_loss = (
                        text_loss * 0.5 +
                        image_loss * 0.3 +
                        reconstruction_loss * mtl_weights.get('reconstruction_weight', 0.2) +
                        coherence_loss * mtl_weights.get('coherence_weight', 0.1)
                    )
                    
                else:
                    # Baseline model
                    image_pred, text_pred = self.model(
                        images, text, text_lengths,
                        teacher_forcing=False
                    )
                    
                    text_loss = self.criterion['text'](
                        text_pred.view(-1, text_pred.size(-1)),
                        targets.view(-1)
                    )
                    image_loss = self.criterion['image'](image_pred, target_images)
                    batch_loss = text_loss * 0.5 + image_loss * 0.5
                
                # Update statistics
                total_loss += batch_loss.item()
                total_text_loss += text_loss.item()
                total_image_loss += image_loss.item()
                
                # Store predictions for metrics
                all_text_predictions.append(text_pred)
                all_text_targets.append(targets)
                all_image_predictions.append(image_pred)
                all_image_targets.append(target_images)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'text_loss': total_text_loss / (batch_idx + 1),
                    'img_loss': total_image_loss / (batch_idx + 1)
                })
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self._calculate_validation_metrics(
            all_text_predictions, all_text_targets,
            all_image_predictions, all_image_targets
        )
        
        return avg_loss, metrics
    
    def _calculate_validation_metrics(self, text_preds, text_targets, 
                                     image_preds, image_targets):
        """Calculate validation metrics"""
        # Concatenate all predictions and targets
        text_preds = torch.cat(text_preds, dim=0)
        text_targets = torch.cat(text_targets, dim=0)
        image_preds = torch.cat(image_preds, dim=0)
        image_targets = torch.cat(image_targets, dim=0)
        
        # Calculate metrics
        metrics = calculate_metrics(
            text_predictions=text_preds,
            text_targets=text_targets,
            image_predictions=image_preds,
            image_targets=image_targets,
            config=self.config
        )
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs=None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        # Early stopping
        patience = self.config['training'].get('early_stopping', {}).get('patience', 15)
        min_delta = self.config['training'].get('early_stopping', {}).get('min_delta', 0.001)
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Metrics: BLEU={val_metrics.get('bleu', 0):.4f}, "
                  f"Perplexity={val_metrics.get('perplexity', 0):.4f}")
            
            # Log to wandb
            if self.config.get('logging', {}).get('wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'val/bleu': val_metrics.get('bleu', 0),
                    'val/perplexity': val_metrics.get('perplexity', 0),
                    'val/cider': val_metrics.get('cider', 0),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint if best model
            if val_loss < self.best_val_loss - min_delta:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = os.path.join(
                    self.config['paths']['checkpoints'],
                    f'best_model_epoch_{epoch+1}.pt'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    metrics=val_metrics,
                    path=checkpoint_path
                )
                print(f"  Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(
                    self.config['paths']['checkpoints'],
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    metrics=val_metrics,
                    path=checkpoint_path
                )
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        final_checkpoint_path = os.path.join(
            self.config['paths']['checkpoints'],
            'final_model.pt'
        )
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            loss=val_loss,
            metrics=val_metrics,
            path=final_checkpoint_path
        )
        
        # Plot training curves
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            save_path=os.path.join(self.config['paths']['results'], 'training_curves.png')
        )
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics_history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch + 1
        }
        
        history_path = os.path.join(self.config['paths']['results'], 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to {final_checkpoint_path}")
        print(f"Training history saved to {history_path}")
        
        return history


def train_baseline(config_path='config.yaml'):
    """Train baseline model"""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['training']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    from data_loader import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config['data']['path'],
        batch_size=config['training']['batch_size'],
        sequence_length=config['data']['sequence_length']
    )
    
    # Initialize baseline model
    model = MultimodalStoryModel(
        image_size=config['model']['image_size'],
        text_vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    # Initialize trainer
    trainer = Trainer(model, config, device)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    return model, history


def train_improved(config_path='config.yaml'):
    """Train improved model with innovations"""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['training']['device'] 
                         if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    from data_loader import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config['data']['path'],
        batch_size=config['training']['batch_size'],
        sequence_length=config['data']['sequence_length']
    )
    
    # Initialize improved model
    model = ImprovedMultimodalStoryModel(
        image_size=config['model']['image_size'],
        text_vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        use_temporal_attention=config['innovation']['temporal_attention'],
        use_cross_modal_fusion=config['innovation']['cross_modal_fusion'],
        num_attention_heads=config['innovation']['attention_heads']
    )
    
    # Initialize trainer
    trainer = Trainer(model, config, device)
    
    # Train with curriculum learning if enabled
    if config['innovation']['curriculum']['enabled']:
        print("Using curriculum learning...")
        
        # Train with increasing sequence lengths
        for seq_len in config['innovation']['curriculum']['sequence_lengths']:
            print(f"\nTraining with sequence length: {seq_len}")
            
            # Create dataloaders with current sequence length
            train_loader_seq, val_loader_seq, _ = create_dataloaders(
                data_path=config['data']['path'],
                batch_size=config['training']['batch_size'],
                sequence_length=seq_len
            )
            
            # Train for specified epochs
            trainer.train(
                train_loader_seq,
                val_loader_seq,
                num_epochs=config['innovation']['curriculum']['epochs_per_length']
            )
    
    # Final training with full sequence length
    print("\nFinal training with full sequence length...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['innovation']['final_epochs']
    )
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multimodal sequence model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['baseline', 'improved'],
                       default='improved', help='Training mode')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda, cpu, mps)')
    
    args = parser.parse_args()
    
    # Update config if device specified
    if args.device:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['training']['device'] = args.device
        with open(args.config, 'w') as f:
            yaml.dump(config, f)
    
    # Train based on mode
    if args.mode == 'baseline':
        print("Training baseline model...")
        model, history = train_baseline(args.config)
    else:
        print("Training improved model...")
        model, history = train_improved(args.config)