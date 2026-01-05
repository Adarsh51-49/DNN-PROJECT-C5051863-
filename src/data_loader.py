"""
Data loading and preprocessing for multimodal sequence modeling
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import pickle
import warnings
warnings.filterwarnings('ignore')


class StoryDataset(Dataset):
    """Dataset for multimodal story sequences"""
    
    def __init__(self, data_path, sequence_length=5, max_text_length=50, 
                 split='train', transform=None, augment=False):
        """
        Args:
            data_path: Path to dataset directory
            sequence_length: Number of frames in each sequence
            max_text_length: Maximum text length
            split: 'train', 'val', or 'test'
            transform: Image transforms
            augment: Whether to use data augmentation
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.max_text_length = max_text_length
        self.split = split
        self.augment = augment
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transform(augment)
        else:
            self.transform = transform
        
        # Build vocabulary
        self.vocab, self.vocab_inv = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Preprocess data
        self.sequences = self._create_sequences()
        
        print(f"Loaded {split} dataset: {len(self.sequences)} sequences")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _load_dataset(self):
        """Load dataset from files"""
        # Check for preprocessed data
        cache_path = self.data_path / f'processed_{self.split}.pkl'
        
        if cache_path.exists():
            print(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Load raw data
        # Assuming dataset structure similar to StoryReasoning
        annotations_path = self.data_path / 'annotations.json'
        
        if not annotations_path.exists():
            # Create dummy data for testing
            print(f"Creating dummy data for testing...")
            data = self._create_dummy_data()
        else:
            with open(annotations_path, 'r') as f:
                data = json.load(f)
        
        # Cache processed data
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        return data
    
    def _create_dummy_data(self):
        """Create dummy data for testing when real data is not available"""
        # This creates synthetic data for testing purposes
        # In practice, you would load the actual StoryReasoning dataset
        
        num_sequences = 1000
        image_dir = self.data_path / 'images'
        image_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy images if they don't exist
        if not any(image_dir.iterdir()):
            print(f"Creating dummy images in {image_dir}...")
            for i in range(num_sequences * self.sequence_length):
                # Create random image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                Image.fromarray(img).save(image_dir / f'image_{i:06d}.jpg')
        
        # Create dummy annotations
        data = []
        story_templates = [
            "A person walking in a park on a sunny day",
            "Children playing with a ball in the garden",
            "A chef preparing food in a restaurant kitchen",
            "A cat sleeping on a comfortable sofa",
            "A car driving on a busy city street",
            "Friends having coffee at a café table",
            "A sunset over mountains with trees",
            "A bird flying in the blue sky",
            "A book on a table with a cup",
            "Rain falling on city buildings"
        ]
        
        for seq_id in range(num_sequences):
            sequence = {
                'sequence_id': seq_id,
                'frames': []
            }
            
            # Choose a story template
            template = story_templates[seq_id % len(story_templates)]
            
            for frame_idx in range(self.sequence_length + 1):  # +1 for target
                frame_data = {
                    'image_path': str(image_dir / f'image_{seq_id*self.sequence_length + frame_idx:06d}.jpg'),
                    'text': f"Frame {frame_idx+1}: {template} with additional details about scene {frame_idx}",
                    'frame_id': frame_idx
                }
                sequence['frames'].append(frame_data)
            
            data.append(sequence)
        
        return data
    
    def _get_default_transform(self, augment=False):
        """Get default image transforms"""
        if augment:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _build_vocabulary(self):
        """Build vocabulary from all text in dataset"""
        # Collect all text
        all_texts = []
        for sequence in self.data:
            for frame in sequence['frames']:
                all_texts.append(frame['text'])
        
        # Tokenize and build vocabulary
        from collections import Counter
        import nltk
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Tokenize all texts
        all_tokens = []
        for text in all_texts:
            tokens = nltk.word_tokenize(text.lower())
            all_tokens.extend(tokens)
        
        # Count tokens
        token_counts = Counter(all_tokens)
        
        # Create vocabulary (most common tokens + special tokens)
        special_tokens = ['<pad>', '<start>', '<end>', '<unk>']
        common_tokens = [token for token, _ in token_counts.most_common(10000 - len(special_tokens))]
        
        vocab_tokens = special_tokens + common_tokens
        vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        vocab_inv = {idx: token for token, idx in vocab.items()}
        
        return vocab, vocab_inv
    
    def _text_to_tokens(self, text):
        """Convert text to token indices"""
        import nltk
        
        # Tokenize
        tokens = nltk.word_tokenize(text.lower())
        
        # Convert to indices
        token_indices = []
        for token in tokens:
            if token in self.vocab:
                token_indices.append(self.vocab[token])
            else:
                token_indices.append(self.vocab['<unk>'])
        
        # Add start and end tokens
        token_indices = [self.vocab['<start>']] + token_indices[:self.max_text_length-2] + [self.vocab['<end>']]
        
        # Pad if needed
        if len(token_indices) < self.max_text_length:
            token_indices = token_indices + [self.vocab['<pad>']] * (self.max_text_length - len(token_indices))
        
        return token_indices[:self.max_text_length]
    
    def _tokens_to_text(self, tokens):
        """Convert token indices back to text"""
        text_tokens = []
        for token_idx in tokens:
            if token_idx == self.vocab['<pad>']:
                break
            if token_idx in self.vocab_inv:
                text_tokens.append(self.vocab_inv[token_idx])
        
        # Remove special tokens
        text_tokens = [t for t in text_tokens if t not in ['<start>', '<end>', '<pad>', '<unk>']]
        
        return ' '.join(text_tokens)
    
    def _create_sequences(self):
        """Create sequences of specified length"""
        sequences = []
        
        for sequence_data in self.data:
            frames = sequence_data['frames']
            
            # Create sliding windows
            for start_idx in range(len(frames) - self.sequence_length):
                # Input sequence
                input_frames = frames[start_idx:start_idx + self.sequence_length]
                
                # Target (next frame)
                target_frame = frames[start_idx + self.sequence_length]
                
                sequences.append({
                    'input_frames': input_frames,
                    'target_frame': target_frame,
                    'sequence_id': sequence_data['sequence_id'],
                    'window_start': start_idx
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Load and transform images
        input_images = []
        for frame in sequence['input_frames']:
            try:
                image = Image.open(frame['image_path']).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                input_images.append(image)
            except:
                # If image loading fails, create a dummy image
                dummy_image = torch.randn(3, 224, 224)
                input_images.append(dummy_image)
        
        # Load target image
        try:
            target_image = Image.open(sequence['target_frame']['image_path']).convert('RGB')
            if self.transform:
                target_image = self.transform(target_image)
        except:
            target_image = torch.randn(3, 224, 224)
        
        # Convert text to tokens
        input_texts = []
        input_text_lengths = []
        
        for frame in sequence['input_frames']:
            tokens = self._text_to_tokens(frame['text'])
            input_texts.append(tokens)
            input_text_lengths.append(len([t for t in tokens if t != self.vocab['<pad>']]))
        
        target_text = self._text_to_tokens(sequence['target_frame']['text'])
        target_text_length = len([t for t in target_text if t != self.vocab['<pad>']])
        
        # Convert to tensors
        input_images = torch.stack(input_images)  # (sequence_length, 3, H, W)
        target_image = target_image  # (3, H, W)
        
        input_texts = torch.tensor(input_texts, dtype=torch.long)  # (sequence_length, max_text_length)
        input_text_lengths = torch.tensor(input_text_lengths, dtype=torch.long)  # (sequence_length,)
        
        target_text = torch.tensor(target_text, dtype=torch.long)  # (max_text_length,)
        
        return {
            'images': input_images,
            'text': input_texts,
            'text_lengths': input_text_lengths,
            'target_images': target_image,
            'target_text': target_text,
            'sequence_id': sequence['sequence_id'],
            'window_start': sequence['window_start'],
            'original_texts': [frame['text'] for frame in sequence['input_frames']] + [sequence['target_frame']['text']]
        }
    
    def get_vocabulary(self):
        """Get vocabulary and inverse vocabulary"""
        return self.vocab, self.vocab_inv
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.vocab_size


def create_dataloaders(data_path, batch_size=32, sequence_length=5, 
                      max_text_length=50, num_workers=4, augment=True):
    """Create train, validation, and test dataloaders"""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = StoryDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        max_text_length=max_text_length,
        split='train',
        transform=train_transform,
        augment=augment
    )
    
    val_dataset = StoryDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        max_text_length=max_text_length,
        split='val',
        transform=val_transform,
        augment=False
    )
    
    test_dataset = StoryDataset(
        data_path=data_path,
        sequence_length=sequence_length,
        max_text_length=max_text_length,
        split='test',
        transform=val_transform,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    # Get max lengths in batch
    max_text_len = max(item['text'].shape[1] for item in batch)
    max_target_len = max(item['target_text'].shape[0] for item in batch)
    
    batch_size = len(batch)
    sequence_length = batch[0]['images'].shape[0]
    
    # Initialize tensors
    images = torch.zeros(batch_size, sequence_length, 3, 224, 224)
    text = torch.zeros(batch_size, sequence_length, max_text_len, dtype=torch.long)
    text_lengths = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    target_images = torch.zeros(batch_size, 3, 224, 224)
    target_text = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    
    # Original texts
    original_texts = []
    
    # Fill tensors
    for i, item in enumerate(batch):
        images[i] = item['images']
        target_images[i] = item['target_images']
        
        # Text and lengths
        curr_text_len = item['text'].shape[1]
        text[i, :, :curr_text_len] = item['text']
        text_lengths[i] = item['text_lengths']
        
        # Target text
        curr_target_len = item['target_text'].shape[0]
        target_text[i, :curr_target_len] = item['target_text']
        
        # Original texts
        original_texts.append(item['original_texts'])
    
    return {
        'images': images,
        'text': text,
        'text_lengths': text_lengths,
        'target_images': target_images,
        'target_text': target_text,
        'original_texts': original_texts
    }


def visualize_batch(batch, vocab_inv=None, num_samples=3):
    """Visualize a batch of data"""
    import matplotlib.pyplot as plt
    
    batch_size = batch['images'].shape[0]
    num_samples = min(num_samples, batch_size)
    
    fig, axes = plt.subplots(num_samples, batch['images'].shape[1] + 1, figsize=(15, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Input images
        for j in range(batch['images'].shape[1]):
            ax = axes[i, j]
            img = batch['images'][i, j].permute(1, 2, 0).cpu().numpy()
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            ax.imshow(img)
            ax.set_title(f'Input Frame {j+1}')
            ax.axis('off')
        
        # Target image
        ax = axes[i, -1]
        target_img = batch['target_images'][i].permute(1, 2, 0).cpu().numpy()
        target_img = target_img * std + mean
        target_img = np.clip(target_img, 0, 1)
        
        ax.imshow(target_img)
        ax.set_title('Target Frame')
        ax.axis('off')
        
        # Add text if vocabulary is provided
        if vocab_inv is not None:
            # Convert token indices to text for the first input frame
            input_text_tokens = batch['text'][i, 0].cpu().numpy()
            input_text = ' '.join([vocab_inv.get(t, '') for t in input_text_tokens 
                                 if t in vocab_inv and vocab_inv[t] not in ['<pad>', '<start>', '<end>']])
            
            target_text_tokens = batch['target_text'][i].cpu().numpy()
            target_text = ' '.join([vocab_inv.get(t, '') for t in target_text_tokens 
                                   if t in vocab_inv and vocab_inv[t] not in ['<pad>', '<start>', '<end>']])
            
            axes[i, 0].text(0.5, -0.1, f'Input: {input_text[:50]}...', 
                           transform=axes[i, 0].transAxes, fontsize=8, ha='center')
            axes[i, -1].text(0.5, -0.1, f'Target: {target_text[:50]}...', 
                            transform=axes[i, -1].transAxes, fontsize=8, ha='center')
    
    plt.suptitle('Batch Visualization', fontsize=16)
    plt.tight_layout()
    plt.show()


def download_dataset(data_path, dataset_url=None):
    """Download dataset if not already present"""
    data_path = Path(data_path)
    
    if data_path.exists() and any(data_path.iterdir()):
        print(f"Dataset already exists at {data_path}")
        return True
    
    # Create directory
    data_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_url:
        print(f"Downloading dataset from {dataset_url}...")
        # Add download logic here
        # For example:
        # import urllib.request
        # urllib.request.urlretrieve(dataset_url, data_path / 'dataset.zip')
        # Then extract
    else:
        print(f"Please download the StoryReasoning dataset manually to {data_path}")
        print("Dataset reference: Oliveira, D. A. P., & Matos, D. M. (2025). "
              "StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding "
              "and Grounded Story Generation. arXiv preprint arXiv:2505.10292")
        print("URL: https://arxiv.org/abs/2505.10292")
    
    return False


# Test the dataset
if __name__ == "__main__":
    print("Testing data loader...")
    
    # Create dummy dataset
    test_data_path = 'test_data'
    
    # Create dataloader
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=test_data_path,
        batch_size=4,
        sequence_length=5,
        max_text_length=30,
        num_workers=0
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Get a batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  images: {batch['images'].shape}")
    print(f"  text: {batch['text'].shape}")
    print(f"  text_lengths: {batch['text_lengths'].shape}")
    print(f"  target_images: {batch['target_images'].shape}")
    print(f"  target_text: {batch['target_text'].shape}")
    
    # Get vocabulary from dataset
    vocab, vocab_inv = train_loader.dataset.get_vocabulary()
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Sample tokens: {list(vocab.keys())[:10]}")
    
    # Visualize a batch
    print("\nVisualizing a batch...")
    visualize_batch(batch, vocab_inv, num_samples=2)
    
    print("\nData loader tested successfully!")