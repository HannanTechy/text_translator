import torch
import torch.nn as nn
import torch.nn.functional as F
import math

vocab = {
    'PAD': 0, 'SOS': 1, 'EOS': 2,
    'i': 3, 'am': 4, 'happy': 5, 'he': 6, 'is': 7, 'sad': 8,
    'je': 9, 'suis': 10, 'heureux': 11, 'il': 12, 'triste': 13
}
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

d_model = 32

embedding = nn.Embedding(vocab_size, d_model)
input_ids = torch.tensor([[1, 3, 4, 5, 2]])  # SOS i am happy EOS

embedded = embedding(input_ids)  #embedding done


# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                                                              # empty matrix for max_len positions
        position = torch.arange(0, max_len).unsqueeze(1)                                               # converting into 2D arranged column
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))             # for freq long and short
        pe[:, 0::2] = torch.sin(position * div_term)                                                   
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

pos_encoder = PositionalEncoding(d_model)
position_encoded = pos_encoder(embedded)  # Shape: [1, 5, 32]
x = position_encoded

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = math.sqrt(d_model)  # Used to scale the dot product (stabilizes gradients)

        # These project the input (x) into Query, Key, and Value matrices
        self.q_proj = nn.Linear(d_model, d_model)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model)  # Value projection

        # Output projection (used after weighted value computation)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)                        #LayerNorm

    def forward(self, x):
        # Create Q, K, V from the input (batch_size, seq_len, d_model)
        Q = self.q_proj(x)  # [1, 5, 32]
        K = self.k_proj(x)  # [1, 5, 32]
        V = self.v_proj(x)  # [1, 5, 32]

        # Compute attention scores (similarity between all Q and K)
        # Q @ K.T → [1, 5, 5] — each token attends to every other token
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply softmax to get attention weights
        # These are probabilities that sum to 1 along each row
        attn_weights = F.softmax(attn_scores, dim=-1)  # [1, 5, 5]

        # Use attention weights to blend (weighted sum) the values V
        # Output is a context-aware representation of each token
        context = torch.matmul(attn_weights, V)  # [1, 5, 32]

        # Final linear transformation of the context
        output = self.out_proj(context)  # [1, 5, 32]
        return output, attn_weights
# Feed-Forward Network block
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        return self.ln(residual + self.ff(x))

# Full Encoder Block = Attention + FFN
class EncoderBlock(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.attn = SelfAttention(d_model)
        self.ffn = FeedForwardBlock(d_model, hidden_dim)

    def forward(self, x):
        x, attn_weights = self.attn(x)
        x = self.ffn(x)
        return x, attn_weights
    
# Instantiate and test
encoder_block = EncoderBlock(d_model, 64)
output, attn_weights = encoder_block(x)

print("Encoder output shape:", output.shape)       # [1, 5, 32]
print("Attention weights shape:", attn_weights.shape)  # [1, 5, 5]
