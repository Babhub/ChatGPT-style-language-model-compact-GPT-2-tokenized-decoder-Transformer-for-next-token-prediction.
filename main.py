# main.py
# Entry point to prepare data, instantiate model, train, and evaluate perplexity on all splits.

import argparse
import torch
from data import GPTTokenizedData
from model import get_best_model_definition
from train import train_model
from evaluation import perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/eval')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Where to save best model')
    args = parser.parse_args()

    # Prepare tokenized data & dataloaders (GPT2 tokenizer provided in data.py)
    tokenized = GPTTokenizedData(batch_size=args.batch_size)
    dataloaders = tokenized.dataloaders  # dict with keys 'train','val','test'
    vocab_size = tokenized.vocab_size
    print("Vocab size:", vocab_size)

    # Instantiate model (model.py)
    model = get_best_model_definition(vocab_size)

    # Train model (train.py)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    train_model(model, {'train': dataloaders['train'], 'val': dataloaders['val']},
                device=device, epochs=args.epochs, lr=args.lr, save_path=args.save_path)

    # Load best model and evaluate perplexity for all three splits using evaluation.perplexity
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    model.to(device)

    print("Evaluating perplexity on train set...")
    ppl_train, _ = perplexity(model, dataloaders['train'])
    print("Train Perplexity:", ppl_train)

    print("Evaluating perplexity on val set...")
    ppl_val, _ = perplexity(model, dataloaders['val'])
    print("Val Perplexity:", ppl_val)

    print("Evaluating perplexity on test set...")
    ppl_test, _ = perplexity(model, dataloaders['test'])
    print("Test Perplexity:", ppl_test)

if __name__ == "__main__":
    main()
