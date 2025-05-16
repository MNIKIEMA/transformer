from loguru import logger
from torch.utils.data import DataLoader
from torch import nn
import torch
import tiktoken
from sacrebleu.metrics import BLEU, CHRF
from src.utils import averager, create_masks


def do_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    tokenizer: tiktoken.Encoding,
    optimizer=None,
):
    """Run a single epoch, either in training or evaluation mode, if `optimizer` is None."""

    device = next(model.parameters()).device
    model.train() if optimizer is not None else model.eval()
    average = averager()
    bleu = BLEU()
    chrf = CHRF()

    for src, tgt in loader:
        print("src", src.shape, "tgt", tgt.shape)
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = create_masks(src, tgt)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(
                src,
                tgt_input,
                #    src_mask=src_mask, tgt_mask=tgt_mask,
            )

            loss = criterion(prediction.reshape(-1, prediction.size(-1)), tgt_output.reshape(-1))
            pred_tokens = prediction.argmax(dim=-1)
            decoded_references = [tokenizer.decode(ref.tolist()).split() for ref in tgt_output]
            decoded_hypotheses = [tokenizer.decode(hyp.tolist()).split() for hyp in pred_tokens]
            references = [" ".join(ref) for ref in decoded_references]
            hypotheses = [" ".join(hyp) for hyp in decoded_hypotheses]

            print("references", references)
            score = bleu.corpus_score(hypotheses=hypotheses, references=references)
            chrf_score = chrf.corpus_score(hypotheses=hypotheses, references=references)

        metrics = {"loss": loss, "bleu": score.score, "chrf": chrf_score.score}
        metrics = average(metrics)

        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    label = "test" if optimizer is None else "train"
    logger.info(
        f"Epoch {epoch:03d} {label: <5} summary "
        f"loss: {metrics['loss']:.3f}, "
        f"acc.: {metrics['bleu']:6.2%}"
    )
    return metrics
