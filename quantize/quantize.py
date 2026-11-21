"""Quantizzazione dinamica int8 per l'inferenza su CPU, con controllo di accordo.

Esempio (Colab o locale):
    python -m quantize.quantize --model /content/drive/MyDrive/llm-local/run1/final --out /content/drive/MyDrive/llm-local/int8

Nota: in XLM-RoBERTa la matrice degli embedding (vocabolario grande) resta in fp32,
quindi la riduzione di dimensione è inferiore al classico 4x.
"""
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

CHECK_SENTENCES = [
    "Il sig. Marco Rossi, codice fiscale RSSMRA85T10A562S, ha presentato la pratica PR-2024/12345.",
    "Il pagamento va effettuato sull'IBAN IT60X0542811101000000123456 intestato a Luca Bianchi.",
    "La riunione è fissata per il 14 novembre alle ore 10.",
]


@torch.no_grad()
def predictions(model, tokenizer, texts: list[str]) -> list[list[int]]:
    out = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        out.append(model(**enc).logits.argmax(-1)[0].tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp32 = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    int8 = torch.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8).eval()

    os.makedirs(args.out, exist_ok=True)
    torch.save(int8, os.path.join(args.out, "model_int8.pt"))  # oggetto intero: caricare con weights_only=False
    tokenizer.save_pretrained(args.out)

    # Controllo: le predizioni int8 devono coincidere con fp32 sulla maggior parte dei token.
    a = predictions(fp32, tokenizer, CHECK_SENTENCES)
    b = predictions(int8, tokenizer, CHECK_SENTENCES)
    same = sum(x == y for ra, rb in zip(a, b) for x, y in zip(ra, rb))
    total = sum(len(r) for r in a)
    print(f"accordo fp32/int8 sui token: {same}/{total} ({same / total:.1%})")
    size_mb = os.path.getsize(os.path.join(args.out, "model_int8.pt")) / 1e6
    print(f"dimensione modello int8: {size_mb:.0f} MB")


if __name__ == "__main__":
    main()
