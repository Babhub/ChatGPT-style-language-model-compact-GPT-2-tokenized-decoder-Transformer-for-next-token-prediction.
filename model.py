# model.py
# Decoder-only Transformer (GPT-style) implementation for HW2.
# - Compatible with evaluation.py which calls `logits = model(inputs, input_padding_mask)`
# - Uses causal mask to prevent future-peeking
# - Sinusoidal positional encodings provided and used
# - get_best_model_definition(vocab_size) returns the model used by evaluation.py
#
# Comments:
# - We accept (input_ids, attention_mask) signature as evaluation.py passes (inputs, input_padding_mask)
# - attention_mask is shape (B, S) with 1 for tokens, 0 for padding.
# - Forward returns logits (B, S, V) when targets is None.
# - If you want to compute loss inside forward, you can pass targets (not needed by evaluation.py).

import math
import torch
from torch import nn
import torch.nn.functional as F

class SinusoidalPositions(nn.Module):
    """
    Provided sinusoidal positional embeddings.
    Adds positional embeddings to token embeddings.
    """
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(-1)  # (S,1)
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier)
        pe[:, 1::2] = torch.cos(position * multiplier)
        # store as buffer so it moves with model.to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, S, D)
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

class TransformerBlock(nn.Module):
    """
    Single Transformer block: LayerNorm -> MultiheadAttention (masked) -> residual
                              -> LayerNorm -> MLP -> residual
    - attn uses batch_first=True so expects (B, S, D) inputs.
    - attn_mask shape: (S, S) or None
    - key_padding_mask shape: (B, S) bool where True indicates positions that should be ignored (PAD)
    """
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # MultiheadAttention handles projections internally
        self.attn = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, batch_first=True)
        # Simple MLP with expansion 4x (common in GPT)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Multi-head attention sublayer with residual
        x_ln = self.ln1(x)
        # attn_mask: (S, S) with -inf where disallowed (we will pass such a mask)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Feedforward sublayer with residual
        x_ln = self.ln2(x)
        x = x + self.mlp(x_ln)
        return x

class GPTLanguageModel(nn.Module):
    """
    Decoder-only GPT-style model.
    Forward signature compatible with evaluation.py:
        logits = model(inputs, input_padding_mask)
    where input_padding_mask is shape (B, S) with 1 for token, 0 for padding.
    """
    def __init__(self,
                 vocab_size,
                 block_size=150,
                 n_layer=6,
                 n_head=8,
                 n_embd=256,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        # token embedding
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # positional encodings (sinusoidal provided)
        self.pos_emb = SinusoidalPositions(block_size, n_embd)

        # stack of transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(n_embd=n_embd, n_head=n_head, dropout=dropout)
                                     for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # initialization
        self._init_weights()

    def _init_weights(self):
        # Xavier init for weights with dims > 1
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len, device):
        # Returns mask of shape (seq_len, seq_len) with 0 where allowed, -inf where masked
        # MultiheadAttention expects additive mask where masked entries are -inf.
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, input_ids, attention_mask=None, targets=None):
        """
        Args:
            input_ids: LongTensor (B, T)
            attention_mask: Byte/Bool/IntTensor (B, T) with 1 for real tokens, 0 for PAD. (evaluation.py passes this)
            targets: optional LongTensor (B, T) with -100 for padding to compute loss inside forward

        Returns:
            logits (B, T, V) if targets is None
            or (logits, loss) if targets provided
        """
        device = input_ids.device
        B, T = input_ids.shape
        if T > self.block_size:
            # trim to last block_size tokens (common in generation/training)
            input_ids = input_ids[:, -self.block_size:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.block_size:]
            T = self.block_size

        # token embeddings + positional encodings
        tok_emb = self.tok_emb(input_ids)       # (B, T, D)
        x = self.pos_emb(tok_emb)               # (B, T, D)

        # prepare masks for attention
        causal = self._causal_mask(T, device)   # (T, T)
        key_padding_mask = None
        if attention_mask is not None:
            # MultiheadAttention expects bool mask where True indicates positions that should be ignored
            # We have attention_mask with 1 for actual tokens; convert to bool for PAD positions
            key_padding_mask = (attention_mask == 0)  # (B, T) bool (True => PAD)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)                         # (B, T, D)
        logits = self.lm_head(x)                 # (B, T, V)

        if targets is None:
            return logits

        # compute loss if requested (targets expected to use -100 for padding)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

def get_best_model_definition(vocab_size):
    """
    This function must be present and return the same model architecture that evaluation.py loads.
    Keep the parameters modest so total trainable parameters < 50M.

    Suggested default:
      - block_size 150 (data tokenized to <=150 in data.py)
      - n_layer 6
      - n_head 8
      - n_embd 256
    These settings yield a model well under 50M parameters for vocab_size ~ 50k.
    """
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=150,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.1
    )
    return model
