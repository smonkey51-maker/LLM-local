"""Fine-tuning di XLM-RoBERTa base per il riconoscimento di dati personali.

Esempio (Colab, GPU T4):
    python -m training.train --data data/synthetic/train.jsonl --out /content/drive/MyDrive/llm-local/run1
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from training.bio import IGNORE, build_label_maps, decode_tokens, evaluate, token_labels


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]


def make_encoder(tokenizer, label2id: dict[str, int], max_len: int):
    def encode(batch: dict) -> dict:
        enc = tokenizer(batch["text"], truncation=True, max_length=max_len,
                        return_offsets_mapping=True)
        labels = [token_labels(offs, spans, label2id)
                  for offs, spans in zip(enc["offset_mapping"], batch["spans"])]
        enc["labels"] = labels
        return enc

    return encode


def make_metrics(id2label: dict[int, str]):
    """Metrica di monitoraggio a livello di token. La valutazione finale a caratteri è in eval/."""

    def compute(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        gold, pred = [], []
        for p_row, l_row in zip(preds, labels):
            keep = l_row != IGNORE
            gold.append(decode_tokens(l_row[keep].tolist(), id2label))
            pred.append(decode_tokens(p_row[keep].tolist(), id2label))
        return {"f1": evaluate(gold, pred)["micro"]["f1"]}

    return compute


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label2id, id2label = build_label_maps()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = load_jsonl(args.data)
    train_rows, val_rows = split(rows, args.val_ratio, args.seed)

    encode = make_encoder(tokenizer, label2id, args.max_len)
    drop = ["text", "spans"]
    train_ds = Dataset.from_list(train_rows).map(encode, batched=True, remove_columns=drop)
    val_ds = Dataset.from_list(val_rows).map(encode, batched=True, remove_columns=drop)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(label2id), id2label=id2label, label2id=label2id)

    targs = TrainingArguments(
        output_dir=f"{args.out}/checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        seed=args.seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_metrics(id2label),
    )
    trainer.train()
    trainer.save_model(f"{args.out}/final")
    tokenizer.save_pretrained(f"{args.out}/final")
    print("salvato in", f"{args.out}/final")


if __name__ == "__main__":
    main()
