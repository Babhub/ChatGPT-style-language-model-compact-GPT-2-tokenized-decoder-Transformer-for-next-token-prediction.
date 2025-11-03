# train.py
# Training loop for the GPTLanguageModel defined in model.py
#
# Key points:
# - Next-token prediction: inputs = tokens[:-1], targets = tokens[1:]
# - Padding tokens produce NO loss: targets replaced with -100 where padding
# - Validation computes average per-token loss and perplexity
# - Saves best model (by validation loss) to save_path

import torch
import torch.nn.functional as F
from torch import nn
import math
import time

def train_model(model, dataloaders, device=None, epochs=8, lr=3e-4, save_path='best_model.pt', grad_clip=1.0):
    """
    dataloaders: dict with keys 'train' and 'val' (and optionally 'test')
    Each dataloader yields dict with 'input_ids' and 'attention_mask'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_epoch = -1

    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val', None)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)            # (B, T')
            attention_mask = batch['attention_mask'].to(device)  # (B, T')

            # prepare inputs/targets
            inputs = input_ids[:, :-1]      # (B, T) - input to model
            targets = input_ids[:, 1:]      # (B, T) - ground truth next tokens
            input_padding_mask = attention_mask[:, :-1]
            target_padding_mask = attention_mask[:, 1:]

            # mask padding tokens in targets to -100 so loss ignores them
            targets_masked = targets.masked_fill(target_padding_mask == 0, -100)

            optimizer.zero_grad()
            logits = model(inputs, input_padding_mask)  # (B, T, V)
            B, S, V = logits.shape
            loss = F.cross_entropy(logits.view(-1, V), targets_masked.view(-1), ignore_index=-100)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # accumulate token-level loss for averaging (multiply avg loss by number of valid tokens)
            valid_tokens = int(target_padding_mask.sum().item())
            epoch_loss_sum += loss.item() * valid_tokens
            epoch_tokens += valid_tokens

        avg_train_loss = epoch_loss_sum / max(1, epoch_tokens)
        t1 = time.time()
        print(f"[Epoch {epoch}] Train loss: {avg_train_loss:.6f}  tokens: {epoch_tokens}  time: {t1-t0:.1f}s")

        # validation
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_tokens = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    inputs = input_ids[:, :-1]
                    targets = input_ids[:, 1:]
                    target_padding_mask = attention_mask[:, 1:]
                    targets_masked = targets.masked_fill(target_padding_mask == 0, -100)

                    logits = model(inputs, attention_mask[:, :-1])
                    B, S, V = logits.shape
                    # use reduction='sum' to get total token loss
                    loss = F.cross_entropy(logits.view(-1, V), targets_masked.view(-1), ignore_index=-100, reduction='sum')
                    val_loss_sum += loss.item()
                    val_tokens += int(target_padding_mask.sum().item())

            avg_val_loss = val_loss_sum / max(1, val_tokens)
            val_ppl = math.exp(avg_val_loss)
            print(f"[Epoch {epoch}] Val loss: {avg_val_loss:.6f}  Val PPL: {val_ppl:.4f}")

            # Save best model by validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path} (epoch {epoch})")

    print(f"Training complete. Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    return
