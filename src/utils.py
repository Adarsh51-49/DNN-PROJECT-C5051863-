"""
Utility functions for multimodal sequence modeling
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import json
import yaml
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, metrics=None, path='checkpoint.pt'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics if metrics else {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer=None, path='checkpoint.pt'):
    """Load model checkpoint"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


def plot_training_curves(train_losses, val_losses, title="Training Curves", save_path=None):
    """Plot training and validation loss curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add min point markers
    min_train_idx = np.argmin(train_losses)
    min_val_idx = np.argmin(val_losses)
    
    ax.scatter(min_train_idx + 1, train_losses[min_train_idx], 
              color='blue', s=100, zorder=5, 
              label=f'Min Train: {train_losses[min_train_idx]:.4f}')
    ax.scatter(min_val_idx + 1, val_losses[min_val_idx], 
              color='red', s=100, zorder=5,
              label=f'Min Val: {val_losses[min_val_idx]:.4f}')
    
    ax.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig


def visualize_attention(attention_weights, sequence, save_path=None):
    """Visualize attention weights"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Heatmap
    sns.heatmap(attention_weights.cpu().numpy(), 
                ax=axes[0], cmap='viridis', 
                xticklabels=sequence, yticklabels=sequence)
    axes[0].set_title('Attention Weights Heatmap')
    axes[0].set_xlabel('Key')
    axes[0].set_ylabel('Query')
    
    # Bar plot for a specific query
    query_idx = len(sequence) // 2
    axes[1].bar(range(len(sequence)), attention_weights[query_idx].cpu().numpy())
    axes[1].set_title(f'Attention Distribution for Query: {sequence[query_idx]}')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_xticks(range(len(sequence)))
    axes[1].set_xticklabels(sequence, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_story(images, texts, predictions=None, save_path=None):
    """Visualize a story sequence with ground truth and predictions"""
    seq_len = len(images)
    
    fig, axes = plt.subplots(2, seq_len, figsize=(5 * seq_len, 10))
    
    for i in range(seq_len):
        # Ground truth image
        ax = axes[0, i] if seq_len > 1 else axes[0]
        if isinstance(images[i], torch.Tensor):
            img = images[i].permute(1, 2, 0).cpu().numpy()
        else:
            img = images[i]
        
        # Normalize image if needed
        if img.max() > 1:
            img = img / 255.0
        
        ax.imshow(img)
        ax.set_title(f'Frame {i+1}\n{texts[i][:50]}...', fontsize=10)
        ax.axis('off')
        
        # Predicted image if provided
        if predictions is not None:
            ax = axes[1, i] if seq_len > 1 else axes[1]
            if isinstance(predictions[i], torch.Tensor):
                pred_img = predictions[i].permute(1, 2, 0).cpu().detach().numpy()
            else:
                pred_img = predictions[i]
            
            # Normalize
            if pred_img.max() > 1:
                pred_img = pred_img / 255.0
            
            ax.imshow(pred_img)
            ax.set_title(f'Prediction {i+1}', fontsize=10)
            ax.axis('off')
    
    plt.suptitle('Story Visualization', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_model_size(model):
    """Compute the size of a model in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device_info():
    """Get information about available devices"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name()
        info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        info['memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
    
    return info


def print_model_summary(model, input_shape):
    """Print model summary similar to Keras"""
    print(f"{'Layer (type)':<30} {'Output Shape':<30} {'Param #':<15}")
    print('=' * 80)
    
    total_params = 0
    trainable_params = 0
    
    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params, trainable_params
            
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f'{class_name}-{module_idx + 1}'
            summary[m_key] = {}
            
            # Output shape
            if isinstance(output, (list, tuple)):
                output_shape = [list(o.shape) for o in output]
            else:
                output_shape = list(output.shape)
            
            summary[m_key]['output_shape'] = output_shape
            
            # Parameters
            params = 0
            for p in module.parameters():
                params += torch.prod(torch.tensor(p.shape))
            
            summary[m_key]['nb_params'] = params
            total_params += params
            
            if hasattr(module, 'weight') and module.weight is not None:
                trainable_params += params if module.weight.requires_grad else 0
        
        if (not isinstance(module, nn.Sequential) and 
            not isinstance(module, nn.ModuleList) and 
            not (module == model)):
            hooks.append(module.register_forward_hook(hook))
    
    # Create dummy input
    if isinstance(input_shape, tuple):
        dummy_input = torch.randn(1, *input_shape)
    else:
        dummy_input = input_shape
    
    # Run hooks
    summary = {}
    hooks = []
    model.apply(register_hook)
    
    # Forward pass
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print summary
    for layer in summary:
        output_shape = str(summary[layer]['output_shape'])
        params = summary[layer]['nb_params']
        
        print(f"{layer:<30} {output_shape:<30} {params:<15,}")
    
    print('=' * 80)
    print(f'Total params: {total_params:,}')
    print(f'Trainable params: {trainable_params:,}')
    print(f'Non-trainable params: {total_params - trainable_params:,}')
    print('=' * 80)
    
    return summary


def create_gradcam_visualization(model, image, target_layer, target_class=None):
    """Create Grad-CAM visualization for model interpretability"""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    # Convert image to appropriate format
    if isinstance(image, torch.Tensor):
        input_tensor = image.unsqueeze(0)
    else:
        input_tensor = torch.tensor(image).unsqueeze(0).float()
    
    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_class)
    
    # Convert image to RGB
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Normalize image
    if image.max() > 1:
        image = image / 255.0
    
    # Overlay CAM on image
    visualization = show_cam_on_image(image, grayscale_cam[0], use_rgb=True)
    
    return visualization


def compute_repetition_rate(text_sequence, ngram_size=3):
    """Compute repetition rate in text sequence"""
    if isinstance(text_sequence, str):
        words = text_sequence.split()
    else:
        words = text_sequence
    
    if len(words) < ngram_size:
        return 0.0
    
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngram = ' '.join(words[i:i + ngram_size])
        ngrams.append(ngram)
    
    unique_ngrams = set(ngrams)
    repetition_rate = 1 - (len(unique_ngrams) / len(ngrams))
    
    return repetition_rate


def analyze_sequence_coherence(sequence, model=None):
    """Analyze coherence of a sequence"""
    coherence_scores = {}
    
    # 1. Semantic similarity between consecutive elements
    from sentence_transformers import SentenceTransformer
    
    if model is None:
        sem_model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        sem_model = model
    
    embeddings = sem_model.encode(sequence)
    
    # Compute pairwise cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)
    
    # Get consecutive similarity
    consecutive_similarities = []
    for i in range(len(sequence) - 1):
        consecutive_similarities.append(similarities[i, i + 1])
    
    coherence_scores['avg_consecutive_similarity'] = np.mean(consecutive_similarities)
    coherence_scores['min_consecutive_similarity'] = np.min(consecutive_similarities)
    
    # 2. Topic consistency
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(sequence)
    
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    topic_distributions = lda.fit_transform(X)
    
    # Compute topic consistency across sequence
    topic_consistency = np.mean(np.std(topic_distributions, axis=0))
    coherence_scores['topic_consistency'] = 1 - topic_consistency  # Lower std = higher consistency
    
    # 3. Narrative flow (simple version based on position)
    position_scores = []
    for i, text in enumerate(sequence):
        # Simple heuristic: later positions should have more complex language
        word_count = len(text.split())
        position_scores.append(word_count / (i + 1))  # Normalize by position
    
    coherence_scores['narrative_flow'] = np.corrcoef(range(len(sequence)), position_scores)[0, 1]
    
    # Overall coherence score (weighted average)
    weights = {
        'avg_consecutive_similarity': 0.4,
        'topic_consistency': 0.3,
        'narrative_flow': 0.3
    }
    
    overall_score = 0
    for key, weight in weights.items():
        score = coherence_scores[key]
        # Normalize score to [0, 1] if needed
        if key == 'narrative_flow':
            score = (score + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        overall_score += score * weight
    
    coherence_scores['overall_coherence'] = overall_score
    
    return coherence_scores


def save_results_to_csv(results_dict, filename='results.csv'):
    """Save results dictionary to CSV file"""
    import pandas as pd
    
    # Convert nested dictionaries to flat structure
    flat_results = {}
    for key, value in results_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_results[f"{key}_{subkey}"] = subvalue
        else:
            flat_results[key] = value
    
    # Create DataFrame and save
    df = pd.DataFrame([flat_results])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path='config.yaml'):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_path}")


def print_config(config, indent=0):
    """Pretty print configuration"""
    for key, value in config.items():
        if isinstance(value, dict):
            print(' ' * indent + f'{key}:')
            print_config(value, indent + 2)
        else:
            print(' ' * indent + f'{key}: {value}')


class ProgressLogger:
    """Progress logger for training"""
    
    def __init__(self, total_steps, desc='Progress'):
        self.total_steps = total_steps
        self.desc = desc
        self.current_step = 0
        self.start_time = datetime.now()
        self.losses = []
        self.metrics = {}
    
    def update(self, loss, **metrics):
        """Update progress with loss and metrics"""
        self.current_step += 1
        self.losses.append(loss)
        
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
        
        # Calculate statistics
        avg_loss = np.mean(self.losses[-100:]) if len(self.losses) > 0 else 0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_second = self.current_step / elapsed if elapsed > 0 else 0
        remaining_time = (self.total_steps - self.current_step) / steps_per_second if steps_per_second > 0 else 0
        
        # Format progress string
        progress_str = (
            f"{self.desc}: {self.current_step}/{self.total_steps} "
            f"Loss: {avg_loss:.4f} "
            f"Time: {elapsed:.1f}s "
            f"ETA: {remaining_time:.1f}s"
        )
        
        # Add metrics
        for key, values in self.metrics.items():
            if values:
                avg_value = np.mean(values[-100:])
                progress_str += f" {key}: {avg_value:.4f}"
        
        print(f"\r{progress_str}", end='')
    
    def finish(self):
        """Finish logging and print summary"""
        print()  # New line
        print(f"Training completed in {(datetime.now() - self.start_time).total_seconds():.1f}s")
        if self.losses:
            print(f"Final average loss: {np.mean(self.losses):.4f}")


# Test utilities
if __name__ == "__main__":
    print("Testing utilities...")
    
    # Set seed
    set_seed(42)
    print("Seed set to 42")
    
    # Get device info
    device_info = get_device_info()
    print("\nDevice Information:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Create dummy model for testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3)
            self.conv2 = nn.Conv2d(16, 32, 3)
            self.fc = nn.Linear(32 * 28 * 28, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    model = DummyModel()
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"\nNumber of parameters: {num_params:,}")
    
    # Compute model size
    model_size = compute_model_size(model)
    print(f"Model size: {model_size:.2f} MB")
    
    # Print model summary
    print("\nModel Summary:")
    print_model_summary(model, (3, 32, 32))
    
    print("\nAll utilities tested successfully!")