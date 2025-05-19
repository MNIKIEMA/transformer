import torch
from tokenizers import Tokenizer, Encoding

class Multi30kDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer: Tokenizer, max_len=50):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_text = self.dataset[idx]["en"]
        tgt_text = self.dataset[idx]["de"]
        src_tokens: Encoding = self.tokenizer.encode(src_text).ids[: self.max_len]
        tgt_tokens: Encoding = self.tokenizer.encode(tgt_text).ids[: self.max_len]
        return {
            "src": torch.tensor(src_tokens, dtype=torch.long),
            "tgt": torch.tensor(tgt_tokens, dtype=torch.long),
        }


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
    def pad_sequence(seqs, pad_id: int = 0):
        max_len = max(seq.size(0) for seq in seqs)
        return torch.stack(
            [
                torch.cat([seq, torch.full((max_len - len(seq),), pad_id, dtype=torch.long)])
                for seq in seqs
            ]
        )

    src_seqs = [item["src"] for item in batch]
    tgt_seqs = [item["tgt"] for item in batch]

    return pad_sequence(src_seqs), pad_sequence(tgt_seqs)
