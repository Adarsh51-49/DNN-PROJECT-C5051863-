"""
Neural Network Architecture for Multimodal Sequence Modeling
Contains both baseline and improved models with innovations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


class VisualEncoder(nn.Module):
    """CNN-based image feature extractor"""
    
    def __init__(self, output_dim=512, pretrained=True):
        super(VisualEncoder, self).__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        modules = list(resnet.children())[:-2]
        self.cnn = nn.Sequential(*modules)
        
        # Adaptive pooling to get fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to output dimension
        self.projection = nn.Linear(2048, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, images):
        """
        Args:
            images: (batch_size, channels, height, width)
        Returns:
            features: (batch_size, output_dim)
        """
        # Extract features
        features = self.cnn(images)  # (batch_size, 2048, 7, 7)
        features = self.adaptive_pool(features)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        
        # Project to desired dimension
        features = self.projection(features)
        features = self.batch_norm(features)
        features = self.dropout(features)
        
        return features


class TextEncoder(nn.Module):
    """LSTM-based text encoder with optional transformer"""
    
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=1024, 
                 num_layers=2, dropout=0.3, bidirectional=True, use_transformer=False):
        super(TextEncoder, self).__init__()
        
        self.use_transformer = use_transformer
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if use_transformer:
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(embedding_dim, hidden_dim * 2 if bidirectional else hidden_dim)
        else:
            # LSTM encoder
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, lengths=None):
        """
        Args:
            text: (batch_size, seq_len)
            lengths: (batch_size,) actual lengths of sequences
        Returns:
            features: (batch_size, hidden_dim * 2) if bidirectional
        """
        # Embed text
        embedded = self.embedding(text)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        if self.use_transformer:
            # Transformer encoding
            if lengths is not None:
                # Create attention mask
                max_len = text.size(1)
                mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
                mask = mask.to(text.device)
            else:
                mask = None
                
            encoded = self.transformer(embedded, src_key_padding_mask=mask)
            
            # Use last non-padded token for each sequence
            if lengths is not None:
                # Get indices of last tokens
                idx = (lengths - 1).view(-1, 1).expand(-1, self.embedding_dim).unsqueeze(1)
                features = encoded.gather(1, idx).squeeze(1)
            else:
                features = encoded[:, -1, :]  # Use last token
            
            features = self.output_proj(features)
        else:
            # LSTM encoding
            if lengths is not None:
                # Pack padded sequence
                packed = pack_padded_sequence(
                    embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed_output, (hidden, cell) = self.lstm(packed)
            else:
                output, (hidden, cell) = self.lstm(embedded)
            
            # Get final hidden states
            if self.lstm.bidirectional:
                # Concatenate forward and backward hidden states
                hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                hidden = hidden[-1]
            
            features = hidden
        
        return features


class TemporalAttention(nn.Module):
    """Temporal-aware attention mechanism with positional encoding"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, max_len=100):
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, hidden_dim))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
        
        # Temporal bias
        self.temporal_bias = nn.Parameter(torch.zeros(num_heads, max_len, max_len))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len, seq_len) optional attention mask
        Returns:
            attended: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Project to query, key, value
        query = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add temporal bias
        scores = scores + self.temporal_bias[:, :seq_len, :seq_len].unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attended = self.output_proj(attended)
        
        return attended, attention_weights


class CrossModalFusion(nn.Module):
    """Gated cross-modal attention fusion"""
    
    def __init__(self, visual_dim, text_dim, hidden_dim, dropout=0.3):
        super(CrossModalFusion, self).__init__()
        
        # Projection layers
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Attention mechanisms
        self.visual_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.text_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        # Gating mechanism
        self.visual_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.text_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, visual_features, text_features, visual_mask=None, text_mask=None):
        """
        Args:
            visual_features: (batch_size, seq_len, visual_dim)
            text_features: (batch_size, seq_len, text_dim)
        Returns:
            fused_features: (batch_size, seq_len, hidden_dim)
        """
        # Project to common space
        visual_proj = self.visual_proj(visual_features)
        text_proj = self.text_proj(text_features)
        
        # Cross-modal attention
        visual_attended, _ = self.visual_attention(
            visual_proj, text_proj, text_proj,
            key_padding_mask=text_mask
        )
        
        text_attended, _ = self.text_attention(
            text_proj, visual_proj, visual_proj,
            key_padding_mask=visual_mask
        )
        
        # Gating mechanism
        visual_gate_input = torch.cat([visual_proj, visual_attended], dim=-1)
        text_gate_input = torch.cat([text_proj, text_attended], dim=-1)
        
        visual_gate = torch.sigmoid(self.visual_gate(visual_gate_input))
        text_gate = torch.sigmoid(self.text_gate(text_gate_input))
        
        # Apply gating
        visual_fused = visual_gate * visual_attended + (1 - visual_gate) * visual_proj
        text_fused = text_gate * text_attended + (1 - text_gate) * text_proj
        
        # Final fusion
        combined = torch.cat([visual_fused, text_fused], dim=-1)
        fused_features = self.fusion(combined)
        
        return fused_features


class SequenceModel(nn.Module):
    """LSTM/GRU-based sequence model for temporal modeling"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, 
                 dropout=0.3, rnn_type='lstm', bidirectional=True):
        super(SequenceModel, self).__init__()
        
        self.rnn_type = rnn_type.lower()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output dimension depends on bidirectional
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None, hidden=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: (batch_size,) actual lengths of sequences
        Returns:
            output: (batch_size, seq_len, output_dim)
            hidden: final hidden state
        """
        if lengths is not None:
            # Pack padded sequence
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.rnn(packed, hidden)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, hidden = self.rnn(x, hidden)
        
        output = self.dropout(output)
        return output, hidden


class ImageDecoder(nn.Module):
    """CNN-based image generator"""
    
    def __init__(self, input_dim, output_channels=3, image_size=224):
        super(ImageDecoder, self).__init__()
        
        self.image_size = image_size
        
        # Determine number of upsampling steps based on image size
        # Assuming we start from 7x7 feature map (like ResNet)
        num_upsample = int(math.log2(image_size // 7))
        
        # Initial projection
        self.fc = nn.Linear(input_dim, 1024 * 7 * 7)
        
        # Transposed convolution layers
        layers = []
        in_channels = 1024
        
        for i in range(num_upsample):
            out_channels = in_channels // 2
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 
                                  kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.3)
            ])
            in_channels = out_channels
        
        # Final layer to get RGB image
        layers.append(
            nn.ConvTranspose2d(in_channels, output_channels, 
                              kernel_size=4, stride=2, padding=1)
        )
        layers.append(nn.Tanh())  # Output in range [-1, 1]
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            image: (batch_size, channels, height, width)
        """
        batch_size = x.size(0)
        
        # Project to spatial dimensions
        x = self.fc(x)
        x = x.view(batch_size, 1024, 7, 7)
        
        # Decode to image
        image = self.decoder(x)
        
        return image


class TextDecoder(nn.Module):
    """LSTM-based text decoder with attention"""
    
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=1024, 
                 num_layers=2, dropout=0.3, attention_dim=512):
        super(TextDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 2,  # embedding + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, prev_hidden, encoder_outputs, mask=None):
        """
        Args:
            input_token: (batch_size, 1) token for current step
            prev_hidden: (num_layers, batch_size, hidden_dim) previous hidden state
            encoder_outputs: (batch_size, seq_len, embedding_dim) encoder outputs
            mask: (batch_size, seq_len) attention mask
        Returns:
            output: (batch_size, vocab_size) logits
            hidden: updated hidden state
            attention_weights: (batch_size, 1, seq_len)
        """
        # Embed input token
        embedded = self.embedding(input_token)  # (batch_size, 1, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Attention over encoder outputs
        attended, attention_weights = self.attention(
            embedded, encoder_outputs, encoder_outputs,
            key_padding_mask=mask
        )
        
        # Concatenate embedded input with attention context
        lstm_input = torch.cat([embedded, attended], dim=-1)
        
        # LSTM step
        output, hidden = self.lstm(lstm_input, prev_hidden)
        output = self.dropout(output)
        
        # Project to vocabulary
        logits = self.fc(output.squeeze(1))
        
        return logits, hidden, attention_weights


class MultimodalStoryModel(nn.Module):
    """Baseline multimodal story model"""
    
    def __init__(self, image_size=224, text_vocab_size=10000, 
                 embedding_dim=512, hidden_dim=1024, num_layers=2, dropout=0.3):
        super(MultimodalStoryModel, self).__init__()
        
        # Encoders
        self.visual_encoder = VisualEncoder(output_dim=embedding_dim)
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Sequence model
        self.sequence_model = SequenceModel(
            input_dim=embedding_dim * 2,  # visual + text
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoders
        self.image_decoder = ImageDecoder(
            input_dim=hidden_dim * 2,  # bidirectional
            image_size=image_size
        )
        
        self.text_decoder = TextDecoder(
            vocab_size=text_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Projection layers
        self.visual_proj = nn.Linear(embedding_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # text encoder is bidirectional
        
    def forward(self, images, text, text_lengths=None, teacher_forcing=False, targets=None):
        """
        Args:
            images: (batch_size, seq_len, channels, height, width)
            text: (batch_size, seq_len, text_len)
            text_lengths: (batch_size, seq_len) actual text lengths
            teacher_forcing: whether to use teacher forcing
            targets: (batch_size, seq_len, text_len) target text for teacher forcing
        Returns:
            image_predictions: (batch_size, seq_len, channels, height, width)
            text_predictions: (batch_size, seq_len, vocab_size, text_len)
        """
        batch_size, seq_len = images.shape[:2]
        
        # Encode each frame
        visual_features = []
        text_features = []
        
        for i in range(seq_len):
            # Encode image
            img_feat = self.visual_encoder(images[:, i])
            visual_features.append(img_feat)
            
            # Encode text
            if text_lengths is not None:
                txt_len = text_lengths[:, i]
            else:
                txt_len = None
                
            txt_feat = self.text_encoder(text[:, i], txt_len)
            text_features.append(txt_feat)
        
        # Stack features
        visual_features = torch.stack(visual_features, dim=1)  # (batch_size, seq_len, embedding_dim)
        text_features = torch.stack(text_features, dim=1)  # (batch_size, seq_len, hidden_dim*2)
        
        # Project to common dimension
        visual_proj = self.visual_proj(visual_features)
        text_proj = self.text_proj(text_features)
        
        # Concatenate modalities
        combined = torch.cat([visual_proj, text_proj], dim=-1)
        
        # Sequence modeling
        sequence_output, _ = self.sequence_model(combined)
        
        # Decode predictions
        image_predictions = []
        text_predictions = []
        
        for i in range(seq_len):
            # Decode image
            img_pred = self.image_decoder(sequence_output[:, i])
            image_predictions.append(img_pred)
            
            # Decode text (autoregressive)
            if teacher_forcing and targets is not None:
                text_pred = self._decode_text_teacher_forcing(
                    sequence_output[:, i], targets[:, i]
                )
            else:
                text_pred = self._decode_text_autoregressive(
                    sequence_output[:, i], max_length=text.shape[2]
                )
            text_predictions.append(text_pred)
        
        image_predictions = torch.stack(image_predictions, dim=1)
        text_predictions = torch.stack(text_predictions, dim=1)
        
        return image_predictions, text_predictions
    
    def _decode_text_teacher_forcing(self, context, target_text):
        """Decode text with teacher forcing"""
        batch_size = context.size(0)
        seq_len = target_text.size(1)
        
        # Initialize
        hidden = None
        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=context.device)
        
        logits_sequence = []
        
        for t in range(seq_len):
            # Decode one step
            logits, hidden, _ = self.text_decoder(
                input_token, hidden, context.unsqueeze(1)
            )
            logits_sequence.append(logits)
            
            # Next input is ground truth
            input_token = target_text[:, t:t+1]
        
        return torch.stack(logits_sequence, dim=1)
    
    def _decode_text_autoregressive(self, context, max_length=50):
        """Decode text autoregressively"""
        batch_size = context.size(0)
        
        # Initialize
        hidden = None
        input_token = torch.zeros(batch_size, 1, dtype=torch.long, device=context.device)
        
        logits_sequence = []
        
        for t in range(max_length):
            # Decode one step
            logits, hidden, _ = self.text_decoder(
                input_token, hidden, context.unsqueeze(1)
            )
            logits_sequence.append(logits)
            
            # Next input is predicted token
            input_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return torch.stack(logits_sequence, dim=1)


class ImprovedMultimodalStoryModel(MultimodalStoryModel):
    """Improved model with temporal-aware cross-modal attention"""
    
    def __init__(self, image_size=224, text_vocab_size=10000, 
                 embedding_dim=512, hidden_dim=1024, num_layers=2, 
                 dropout=0.3, use_temporal_attention=True, 
                 use_cross_modal_fusion=True, num_attention_heads=8):
        super().__init__(image_size, text_vocab_size, embedding_dim, 
                        hidden_dim, num_layers, dropout)
        
        self.use_temporal_attention = use_temporal_attention
        self.use_cross_modal_fusion = use_cross_modal_fusion
        
        # Temporal attention
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=hidden_dim * 2,  # sequence model output dimension
                num_heads=num_attention_heads,
                dropout=dropout
            )
        
        # Cross-modal fusion
        if use_cross_modal_fusion:
            self.cross_modal_fusion = CrossModalFusion(
                visual_dim=embedding_dim,
                text_dim=hidden_dim * 2,  # text encoder output dimension
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            
            # Update sequence model input dimension
            self.sequence_model = SequenceModel(
                input_dim=hidden_dim,  # after fusion
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # Multi-task learning heads
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim * 2)  # reconstruct input features
        )
        
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # coherence score between 0 and 1
        )
        
    def forward(self, images, text, text_lengths=None, teacher_forcing=False, 
                targets=None, return_intermediate=False):
        """
        Enhanced forward pass with innovations
        """
        batch_size, seq_len = images.shape[:2]
        
        # Encode each frame (same as baseline)
        visual_features = []
        text_features = []
        
        for i in range(seq_len):
            img_feat = self.visual_encoder(images[:, i])
            visual_features.append(img_feat)
            
            if text_lengths is not None:
                txt_len = text_lengths[:, i]
            else:
                txt_len = None
                
            txt_feat = self.text_encoder(text[:, i], txt_len)
            text_features.append(txt_feat)
        
        visual_features = torch.stack(visual_features, dim=1)
        text_features = torch.stack(text_features, dim=1)
        
        # Apply cross-modal fusion if enabled
        if self.use_cross_modal_fusion:
            # Create masks if lengths are provided
            visual_mask = None
            text_mask = None
            
            if text_lengths is not None:
                # Create text mask (1 for valid, 0 for padding)
                max_len = text.shape[2]
                text_mask = torch.arange(max_len).expand(batch_size, max_len) < text_lengths[:, :, None]
                text_mask = text_mask.any(dim=1)  # Combine across sequence
                text_mask = ~text_mask  # Invert for attention mask
            
            fused_features = self.cross_modal_fusion(
                visual_features, text_features, visual_mask, text_mask
            )
            combined = fused_features
        else:
            # Baseline combination
            visual_proj = self.visual_proj(visual_features)
            text_proj = self.text_proj(text_features)
            combined = torch.cat([visual_proj, text_proj], dim=-1)
        
        # Apply temporal attention if enabled
        if self.use_temporal_attention:
            # Create causal mask for autoregressive generation
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(combined.device)
            
            attended, attention_weights = self.temporal_attention(
                combined, mask=causal_mask.unsqueeze(0)
            )
            sequence_input = attended
        else:
            sequence_input = combined
            attention_weights = None
        
        # Sequence modeling
        sequence_output, _ = self.sequence_model(sequence_input)
        
        # Multi-task learning outputs
        reconstruction_output = self.reconstruction_head(sequence_output)
        coherence_scores = self.coherence_head(sequence_output)
        
        # Decode predictions
        image_predictions = []
        text_predictions = []
        
        for i in range(seq_len):
            img_pred = self.image_decoder(sequence_output[:, i])
            image_predictions.append(img_pred)
            
            if teacher_forcing and targets is not None:
                text_pred = self._decode_text_teacher_forcing(
                    sequence_output[:, i], targets[:, i]
                )
            else:
                text_pred = self._decode_text_autoregressive(
                    sequence_output[:, i], max_length=text.shape[2]
                )
            text_predictions.append(text_pred)
        
        image_predictions = torch.stack(image_predictions, dim=1)
        text_predictions = torch.stack(text_predictions, dim=1)
        
        if return_intermediate:
            return {
                'image_predictions': image_predictions,
                'text_predictions': text_predictions,
                'visual_features': visual_features,
                'text_features': text_features,
                'attention_weights': attention_weights,
                'reconstruction_output': reconstruction_output,
                'coherence_scores': coherence_scores
            }
        
        return image_predictions, text_predictions


# Example usage and testing
if __name__ == "__main__":
    # Test baseline model
    print("Testing Baseline Model...")
    baseline_model = MultimodalStoryModel()
    
    batch_size = 4
    seq_len = 5
    text_len = 20
    
    # Create dummy inputs
    images = torch.randn(batch_size, seq_len, 3, 224, 224)
    text = torch.randint(0, 10000, (batch_size, seq_len, text_len))
    text_lengths = torch.randint(10, text_len, (batch_size, seq_len))
    
    # Forward pass
    image_pred, text_pred = baseline_model(images, text, text_lengths)
    
    print(f"Input images shape: {images.shape}")
    print(f"Input text shape: {text.shape}")
    print(f"Output image predictions shape: {image_pred.shape}")
    print(f"Output text predictions shape: {text_pred.shape}")
    
    # Test improved model
    print("\nTesting Improved Model...")
    improved_model = ImprovedMultimodalStoryModel(
        use_temporal_attention=True,
        use_cross_modal_fusion=True
    )
    
    outputs = improved_model(images, text, text_lengths, return_intermediate=True)
    
    print(f"Image predictions shape: {outputs['image_predictions'].shape}")
    print(f"Text predictions shape: {outputs['text_predictions'].shape}")
    print(f"Attention weights shape: {outputs['attention_weights'].shape}")
    print(f"Coherence scores shape: {outputs['coherence_scores'].shape}")
    
    print("\nAll models initialized successfully!")