"""
Evaluation metrics and functions for multimodal sequence modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')


def calculate_perplexity(logits, targets, ignore_index=0):
    """Calculate perplexity from logits and targets"""
    # Flatten logits and targets
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    # Create mask for valid tokens
    mask = targets_flat != ignore_index
    
    # Calculate cross entropy loss
    loss = F.cross_entropy(logits_flat[mask], targets_flat[mask], reduction='mean')
    
    # Perplexity is exp(loss)
    perplexity = torch.exp(loss)
    
    return perplexity.item()


def calculate_bleu(predictions, targets, weights=(0.25, 0.25, 0.25, 0.25)):
    """Calculate BLEU score for text generation"""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Convert token indices to words (simplified)
    # In practice, you would use a vocabulary mapping
    bleu_scores = []
    smoothie = SmoothingFunction().method1
    
    for pred, target in zip(predictions, targets):
        # Convert to strings (simplified - use actual vocabulary)
        pred_str = ' '.join([str(p) for p in pred])
        target_str = ' '.join([str(t) for t in target])
        
        # Tokenize
        pred_tokens = nltk.word_tokenize(pred_str)
        target_tokens = [nltk.word_tokenize(target_str)]
        
        # Calculate BLEU
        try:
            score = sentence_bleu(
                target_tokens, 
                pred_tokens, 
                weights=weights,
                smoothing_function=smoothie
            )
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)
    
    return np.mean(bleu_scores)


def calculate_rouge(predictions, targets):
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, target in zip(predictions, targets):
        # Convert to strings
        pred_str = ' '.join([str(p) for p in pred])
        target_str = ' '.join([str(t) for t in target])
        
        scores = scorer.score(target_str, pred_str)
        
        for key in rouge_scores.keys():
            rouge_scores[key].append(scores[key].fmeasure)
    
    # Average scores
    avg_scores = {key: np.mean(values) for key, values in rouge_scores.items()}
    return avg_scores


def calculate_meteor(predictions, targets):
    """Calculate METEOR score"""
    meteor_scores = []
    
    for pred, target in zip(predictions, targets):
        # Convert to strings
        pred_str = ' '.join([str(p) for p in pred])
        target_str = ' '.join([str(t) for t in target])
        
        try:
            score = meteor_score([target_str], pred_str)
            meteor_scores.append(score)
        except:
            meteor_scores.append(0.0)
    
    return np.mean(meteor_scores)


def calculate_cider(predictions, targets):
    """Calculate CIDEr score (simplified implementation)"""
    # Note: For full CIDEr implementation, use pycocoevalcap
    # This is a simplified version
    
    def get_n_grams(tokens, n):
        """Get n-grams from tokens"""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    cider_scores = []
    
    for pred, target in zip(predictions, targets):
        # Convert to strings and tokenize
        pred_str = ' '.join([str(p) for p in pred])
        target_str = ' '.join([str(t) for t in target])
        
        pred_tokens = nltk.word_tokenize(pred_str.lower())
        target_tokens = nltk.word_tokenize(target_str.lower())
        
        # Calculate n-gram precision and recall for n=1 to 4
        precisions = []
        recalls = []
        
        for n in range(1, 5):
            pred_ngrams = get_n_grams(pred_tokens, n)
            target_ngrams = get_n_grams(target_tokens, n)
            
            if not pred_ngrams or not target_ngrams:
                precisions.append(0)
                recalls.append(0)
                continue
            
            # Count matches
            pred_counter = Counter(pred_ngrams)
            target_counter = Counter(target_ngrams)
            
            matches = sum((pred_counter & target_counter).values())
            
            precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0
            recall = matches / len(target_ngrams) if len(target_ngrams) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate geometric mean of F-scores
        f_scores = []
        for p, r in zip(precisions, recalls):
            if p + r > 0:
                f_scores.append(2 * p * r / (p + r))
            else:
                f_scores.append(0)
        
        # Average F-scores (simplified CIDEr)
        cider_score = np.mean(f_scores) if f_scores else 0
        cider_scores.append(cider_score)
    
    return np.mean(cider_scores)


def calculate_image_metrics(predictions, targets):
    """Calculate image quality metrics"""
    # Convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    metrics = {}
    
    # MSE (already used in loss)
    mse = np.mean((predictions - targets) ** 2)
    metrics['mse'] = float(mse)
    
    # PSNR
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        metrics['psnr'] = float(psnr)
    else:
        metrics['psnr'] = float('inf')
    
    # SSIM (simplified calculation)
    # Note: For proper SSIM, use skimage.metrics.structural_similarity
    metrics['ssim'] = 0.0  # Placeholder
    
    return metrics


def calculate_repetition_rate(predictions, ngram_size=3):
    """Calculate repetition rate in generated text"""
    repetition_rates = []
    
    for pred in predictions:
        # Convert to string
        if isinstance(pred, (list, np.ndarray)):
            pred_str = ' '.join([str(p) for p in pred])
        else:
            pred_str = str(pred)
        
        # Tokenize
        tokens = nltk.word_tokenize(pred_str.lower())
        
        if len(tokens) < ngram_size:
            repetition_rates.append(0.0)
            continue
        
        # Count n-grams
        ngrams = []
        for i in range(len(tokens) - ngram_size + 1):
            ngram = ' '.join(tokens[i:i+ngram_size])
            ngrams.append(ngram)
        
        # Calculate repetition rate
        unique_ngrams = set(ngrams)
        repetition_rate = 1 - (len(unique_ngrams) / len(ngrams))
        repetition_rates.append(repetition_rate)
    
    return np.mean(repetition_rates)


def calculate_coherence_score(predictions, targets=None):
    """Calculate coherence score for generated sequences"""
    # This is a simplified coherence score
    # In practice, you might use more sophisticated methods
    
    coherence_scores = []
    
    for pred_seq in predictions:
        if len(pred_seq) < 2:
            coherence_scores.append(0.0)
            continue
        
        # Convert each prediction in sequence to string
        pred_strs = []
        for pred in pred_seq:
            if isinstance(pred, (list, np.ndarray)):
                pred_str = ' '.join([str(p) for p in pred])
            else:
                pred_str = str(pred)
            pred_strs.append(pred_str)
        
        # Calculate semantic similarity between consecutive elements
        similarities = []
        for i in range(len(pred_strs) - 1):
            # Simplified similarity: Jaccard similarity of words
            words1 = set(nltk.word_tokenize(pred_strs[i].lower()))
            words2 = set(nltk.word_tokenize(pred_strs[i+1].lower()))
            
            if len(words1) == 0 or len(words2) == 0:
                similarity = 0.0
            else:
                similarity = len(words1 & words2) / len(words1 | words2)
            
            similarities.append(similarity)
        
        # Average similarity as coherence score
        coherence_score = np.mean(similarities) if similarities else 0.0
        coherence_scores.append(coherence_score)
    
    return np.mean(coherence_scores)


def calculate_metrics(text_predictions, text_targets, image_predictions=None, 
                     image_targets=None, config=None):
    """Calculate all metrics"""
    metrics = {}
    
    # Text metrics
    if text_predictions is not None and text_targets is not None:
        # Perplexity
        if isinstance(text_predictions, torch.Tensor) and isinstance(text_targets, torch.Tensor):
            metrics['perplexity'] = calculate_perplexity(text_predictions, text_targets)
        
        # Convert to numpy for other metrics
        if isinstance(text_predictions, torch.Tensor):
            text_pred_np = text_predictions.argmax(dim=-1).cpu().numpy()
        else:
            text_pred_np = text_predictions
        
        if isinstance(text_targets, torch.Tensor):
            text_target_np = text_targets.cpu().numpy()
        else:
            text_target_np = text_targets
        
        # BLEU scores
        metrics['bleu1'] = calculate_bleu(text_pred_np, text_target_np, weights=(1, 0, 0, 0))
        metrics['bleu2'] = calculate_bleu(text_pred_np, text_target_np, weights=(0.5, 0.5, 0, 0))
        metrics['bleu3'] = calculate_bleu(text_pred_np, text_target_np, weights=(0.33, 0.33, 0.33, 0))
        metrics['bleu4'] = calculate_bleu(text_pred_np, text_target_np, weights=(0.25, 0.25, 0.25, 0.25))
        metrics['bleu'] = metrics['bleu4']  # Main BLEU score
        
        # ROUGE scores
        rouge_scores = calculate_rouge(text_pred_np, text_target_np)
        metrics.update(rouge_scores)
        
        # METEOR
        metrics['meteor'] = calculate_meteor(text_pred_np, text_target_np)
        
        # CIDEr
        metrics['cider'] = calculate_cider(text_pred_np, text_target_np)
        
        # Repetition rate
        metrics['repetition_rate'] = calculate_repetition_rate(text_pred_np)
        
        # Coherence score
        metrics['coherence_score'] = calculate_coherence_score(text_pred_np)
    
    # Image metrics
    if image_predictions is not None and image_targets is not None:
        image_metrics = calculate_image_metrics(image_predictions, image_targets)
        metrics.update(image_metrics)
    
    # Human evaluation simulation
    if config and config.get('evaluation', {}).get('human_evaluation', {}).get('simulate', False):
        metrics['human_evaluation'] = simulate_human_evaluation(metrics, config)
    
    return metrics


def simulate_human_evaluation(metrics, config):
    """Simulate human evaluation scores based on metrics"""
    # Weights for different metrics in human evaluation
    weights = config['evaluation']['human_evaluation'].get('weights', {
        'bleu': 0.3,
        'perplexity': 0.2,
        'cider': 0.2,
        'coherence_score': 0.3
    })
    
    # Normalize metrics to [0, 1] range
    normalized_scores = {}
    
    # BLEU (already in [0, 1])
    normalized_scores['bleu'] = metrics.get('bleu', 0)
    
    # Perplexity: lower is better, normalize with sigmoid
    perplexity = metrics.get('perplexity', 50)
    normalized_scores['perplexity'] = 1 / (1 + np.exp((perplexity - 20) / 10))
    
    # CIDEr (already in [0, 1])
    normalized_scores['cider'] = metrics.get('cider', 0)
    
    # Coherence score (already in [0, 1])
    normalized_scores['coherence_score'] = metrics.get('coherence_score', 0)
    
    # Calculate weighted score
    human_score = 0
    for metric, weight in weights.items():
        if metric in normalized_scores:
            human_score += normalized_scores[metric] * weight
    
    # Scale to 1-5 range
    human_score_scaled = 1 + 4 * human_score
    
    # Add some random noise to simulate human variance
    noise = np.random.normal(0, 0.2)
    human_score_scaled = np.clip(human_score_scaled + noise, 1, 5)
    
    return human_score_scaled


def generate_story(model, sequence, device='cuda', max_length=50, temperature=0.8, top_k=50):
    """Generate story continuation"""
    model.eval()
    
    with torch.no_grad():
        # Extract input sequence
        images = sequence['images'].to(device)
        text = sequence['text'].to(device)
        text_lengths = sequence['text_lengths'].to(device)
        
        # Forward pass
        if hasattr(model, 'use_temporal_attention') and model.use_temporal_attention:
            outputs = model(
                images, text, text_lengths,
                teacher_forcing=False,
                return_intermediate=True
            )
            
            # Get attention weights for visualization
            attention_weights = outputs.get('attention_weights', None)
        else:
            image_pred, text_pred = model(
                images, text, text_lengths,
                teacher_forcing=False
            )
            outputs = {
                'image_predictions': image_pred,
                'text_predictions': text_pred
            }
            attention_weights = None
        
        # Generate text for each step
        generated_texts = []
        
        for i in range(text_pred.shape[1]):  # For each step in sequence
            # Get logits for this step
            step_logits = text_pred[:, i, :, :]  # (batch, vocab_size, seq_len)
            
            # Apply temperature sampling
            step_logits = step_logits / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = step_logits < torch.topk(step_logits, top_k)[0][..., -1, None]
                step_logits[indices_to_remove] = -float('inf')
            
            # Apply softmax and sample
            probs = F.softmax(step_logits, dim=-1)
            
            # Sample from distribution
            sampled_indices = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            sampled_indices = sampled_indices.view(probs.size(0), probs.size(1))
            
            # Convert indices to text (simplified)
            # In practice, use vocabulary to convert to words
            generated_texts.append(sampled_indices.cpu().numpy())
        
        # Convert image predictions to numpy
        image_predictions = outputs['image_predictions'].cpu().numpy()
        
        return {
            'image_predictions': image_predictions,
            'text_predictions': generated_texts,
            'attention_weights': attention_weights
        }


def run_ablation_study(model_class, config, train_loader, val_loader, test_loader, 
                      ablation_configs, device='cuda'):
    """Run ablation study with different model configurations"""
    results = {}
    
    for config_name, model_config in ablation_configs.items():
        print(f"\nRunning ablation: {config_name}")
        print(f"Configuration: {model_config}")
        
        # Create model with specific configuration
        model = model_class(**model_config).to(device)
        
        # Train model
        from train import Trainer
        trainer = Trainer(model, config, device)
        
        # Train for fewer epochs for ablation study
        history = trainer.train(train_loader, val_loader, num_epochs=10)
        
        # Evaluate on test set
        _, test_metrics = trainer.validate_epoch(test_loader)
        
        # Store results
        results[config_name] = {
            'config': model_config,
            'final_val_loss': history['val_losses'][-1],
            'best_val_loss': history['best_val_loss'],
            'test_metrics': test_metrics
        }
        
        # Save model
        checkpoint_path = f"checkpoints/ablation_{config_name}.pt"
        torch.save(model.state_dict(), checkpoint_path)
    
    return results


def create_comparison_table(baseline_results, improved_results, save_path=None):
    """Create comparison table between baseline and improved results"""
    import pandas as pd
    
    # Extract metrics
    comparison_data = []
    
    # Define metrics to compare
    metrics_to_compare = [
        'bleu', 'bleu1', 'bleu2', 'bleu3', 'bleu4',
        'rouge1', 'rouge2', 'rougeL',
        'meteor', 'cider', 'perplexity',
        'repetition_rate', 'coherence_score',
        'mse', 'psnr'
    ]
    
    for metric in metrics_to_compare:
        baseline_val = baseline_results.get(metric, 0)
        improved_val = improved_results.get(metric, 0)
        
        # Calculate improvement
        if baseline_val != 0:
            improvement = ((improved_val - baseline_val) / baseline_val) * 100
        else:
            improvement = 0
        
        comparison_data.append({
            'Metric': metric.upper(),
            'Baseline': f"{baseline_val:.4f}",
            'Improved': f"{improved_val:.4f}",
            'Improvement (%)': f"{improvement:+.2f}%"
        })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to file if path provided
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to {save_path}")
    
    return df


def plot_metrics_comparison(baseline_metrics, improved_metrics, save_path=None):
    """Plot comparison of metrics between baseline and improved models"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Select metrics for visualization
    text_metrics = ['bleu', 'rouge1', 'meteor', 'cider']
    quality_metrics = ['perplexity', 'repetition_rate', 'coherence_score']
    
    # Plot 1: Text generation metrics
    x = np.arange(len(text_metrics))
    width = 0.35
    
    baseline_vals = [baseline_metrics.get(m, 0) for m in text_metrics]
    improved_vals = [improved_metrics.get(m, 0) for m in text_metrics]
    
    axes[0, 0].bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    axes[0, 0].bar(x + width/2, improved_vals, width, label='Improved', alpha=0.8)
    axes[0, 0].set_xlabel('Metric')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Text Generation Metrics')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(text_metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Quality metrics (lower is better for some)
    x = np.arange(len(quality_metrics))
    
    baseline_vals = [baseline_metrics.get(m, 0) for m in quality_metrics]
    improved_vals = [improved_metrics.get(m, 0) for m in quality_metrics]
    
    axes[0, 1].bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    axes[0, 1].bar(x + width/2, improved_vals, width, label='Improved', alpha=0.8)
    axes[0, 1].set_xlabel('Metric')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Quality Metrics')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(quality_metrics)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Improvement percentages
    improvements = []
    for metric in text_metrics + quality_metrics:
        baseline = baseline_metrics.get(metric, 0)
        improved = improved_metrics.get(metric, 0)
        
        if baseline != 0:
            improvement = ((improved - baseline) / baseline) * 100
        else:
            improvement = 0
        
        improvements.append(improvement)
    
    x = np.arange(len(text_metrics + quality_metrics))
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    axes[1, 0].bar(x, improvements, color=colors, alpha=0.8)
    axes[1, 0].set_xlabel('Metric')
    axes[1, 0].set_ylabel('Improvement (%)')
    axes[1, 0].set_title('Percentage Improvement')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(text_metrics + quality_metrics, rotation=45, ha='right')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Attention visualization (if available)
    if 'attention_weights' in baseline_metrics and 'attention_weights' in improved_metrics:
        # Sample attention weights
        baseline_attn = baseline_metrics['attention_weights'][0].cpu().numpy()
        improved_attn = improved_metrics['attention_weights'][0].cpu().numpy()
        
        im1 = axes[1, 1].imshow(baseline_attn, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Baseline Attention')
        axes[1, 1].set_xlabel('Key Position')
        axes[1, 1].set_ylabel('Query Position')
        plt.colorbar(im1, ax=axes[1, 1])
    
    plt.suptitle('Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
    return fig


def save_evaluation_results(results, filename='evaluation_results.json'):
    """Save evaluation results to JSON file"""
    # Convert numpy values to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    results = convert_numpy(results)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {filename}")


def load_evaluation_results(filename='evaluation_results.json'):
    """Load evaluation results from JSON file"""
    with open(filename, 'r') as f:
        results = json.load(f)
    
    print(f"Evaluation results loaded from {filename}")
    return results


# Example usage
if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Create dummy data
    batch_size = 4
    seq_len = 5
    vocab_size = 10000
    text_len = 20
    
    # Dummy predictions and targets
    text_predictions = torch.randn(batch_size, seq_len, vocab_size, text_len)
    text_targets = torch.randint(0, vocab_size, (batch_size, seq_len, text_len))
    
    image_predictions = torch.randn(batch_size, seq_len, 3, 224, 224)
    image_targets = torch.randn(batch_size, seq_len, 3, 224, 224)
    
    # Calculate metrics
    metrics = calculate_metrics(
        text_predictions=text_predictions,
        text_targets=text_targets,
        image_predictions=image_predictions,
        image_targets=image_targets
    )
    
    print("\nCalculated Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test repetition rate
    dummy_texts = [
        "the cat sat on the mat the cat was happy",
        "once upon a time in a faraway land",
        "hello world hello world hello world"
    ]
    
    rep_rate = calculate_repetition_rate(dummy_texts)
    print(f"\nRepetition rate for dummy texts: {rep_rate:.4f}")
    
    # Test coherence score
    coherence_score = calculate_coherence_score([dummy_texts])
    print(f"Coherence score for dummy sequence: {coherence_score:.4f}")
    
    print("\nAll evaluation functions tested successfully!")