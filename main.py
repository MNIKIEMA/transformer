import tiktoken
import torch
from torch.utils.data import DataLoader
from torch import nn
from datasets import load_dataset
from src.data import Multi30kDataset, collate_fn
from src.transformer import Transformer
from src.train import do_epoch


def main():
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load datasets
    train_ds = load_dataset("bentrevett/multi30k", split="train").take(10)  # type: ignore
    val_ds = load_dataset("bentrevett/multi30k", split="validation").take(5)  # type: ignore

    # Build datasets + loaders
    train_dataset = Multi30kDataset(train_ds, tokenizer, max_len=50)

    val_dataset = Multi30kDataset(val_ds, tokenizer, max_len=50)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Model
    vocab_size = tokenizer.n_vocab
    model = Transformer(
        enc_vocab_size=vocab_size,
        dec_vocab_size=vocab_size,
        d_ffn=2048,
        d_model=512,
        n_heads=8,
        num_layers=6,
    ).to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train loop
    epochs = 5
    for epoch in range(epochs):
        train_loss = do_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            tokenizer=tokenizer,
            epoch=epoch,
        )
        val_loss = do_epoch(
            model=model, criterion=criterion, loader=val_loader, tokenizer=tokenizer, epoch=epoch
        )
        print(
            f"Epoch {epoch + 1} / {epochs}: Train loss {train_loss:.4f} | Val loss {val_loss:.4f}"
        )
