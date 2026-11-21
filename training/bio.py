"""Conversione tra span a caratteri e etichette BIO a livello di token.

Nessuna dipendenza da Hugging Face: il modulo è testabile senza scaricare modelli.
"""
from __future__ import annotations

IGNORE = -100  # indice ignorato dalla loss (token speciali, padding)

TAGS = ["FULLNAME", "CF", "PIVA", "IBAN", "CARD", "EMAIL", "PEC", "PHONE",
        "ADDRESS", "PRATICA", "TARGA", "CATASTO", "URL"]


def build_label_maps(tags: list[str] = TAGS) -> tuple[dict[str, int], dict[int, str]]:
    labels = ["O"] + [f"{p}-{t}" for t in tags for p in ("B", "I")]
    label2id = {lab: i for i, lab in enumerate(labels)}
    return label2id, {i: lab for lab, i in label2id.items()}


def token_labels(offsets: list[tuple[int, int]], spans: list[dict],
                 label2id: dict[str, int]) -> list[int]:
    """Assegna un'etichetta BIO a ogni token dato il suo intervallo di caratteri.

    Il primo token di uno span riceve B-TAG, i successivi I-TAG.
    I token con offset vuoto (speciali) ricevono IGNORE.
    """
    labels: list[int] = []
    prev_hit: int | None = None
    for start, end in offsets:
        if start == end:
            labels.append(IGNORE)
            prev_hit = None
            continue
        hit = next((i for i, sp in enumerate(spans)
                    if start < sp["end"] and end > sp["start"]), None)
        if hit is None:
            labels.append(label2id["O"])
            prev_hit = None
            continue
        prefix = "I" if prev_hit == hit else "B"
        labels.append(label2id[f"{prefix}-{spans[hit]['tag']}"])
        prev_hit = hit
    return labels


def _iter_chunks(label_ids: list[int], id2label: dict[int, str]):
    """Yield (tag, i, j) con indici di posizione inclusivi, saltando IGNORE."""
    cur = None
    for pos, lid in enumerate(label_ids):
        if lid == IGNORE:
            continue
        lab = id2label[lid]
        if lab == "O":
            if cur:
                yield cur
            cur = None
            continue
        prefix, tag = lab.split("-", 1)
        if prefix == "B" or cur is None or cur[0] != tag:
            if cur:
                yield cur
            cur = (tag, pos, pos)
        else:
            cur = (tag, cur[1], pos)
    if cur:
        yield cur


def decode_tokens(label_ids: list[int], id2label: dict[int, str]) -> set[tuple[str, int, int]]:
    """Entità come (tag, primo_token, ultimo_token), sugli indici della sequenza filtrata."""
    return set(_iter_chunks(label_ids, id2label))


def decode_chars(offsets: list[tuple[int, int]], label_ids: list[int],
                 id2label: dict[int, str]) -> list[dict]:
    """Converte le etichette predette in span a caratteri {tag, start, end}."""
    out = []
    for tag, i, j in _iter_chunks(label_ids, id2label):
        out.append({"tag": tag, "start": offsets[i][0], "end": offsets[j][1]})
    return out


def evaluate(gold: list[set], pred: list[set]) -> dict:
    """Precisione, richiamo e F1 micro e per tag, su insiemi di entità per esempio.

    Le entità sono confrontate con match esatto di tag e confini.
    """
    per_tag: dict[str, dict[str, int]] = {}

    def bump(tag: str, key: str) -> None:
        per_tag.setdefault(tag, {"tp": 0, "fp": 0, "fn": 0})[key] += 1

    for g, p in zip(gold, pred):
        for ent in g & p:
            bump(ent[0], "tp")
        for ent in p - g:
            bump(ent[0], "fp")
        for ent in g - p:
            bump(ent[0], "fn")

    def prf(tp: int, fp: int, fn: int) -> dict:
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    tp = sum(c["tp"] for c in per_tag.values())
    fp = sum(c["fp"] for c in per_tag.values())
    fn = sum(c["fn"] for c in per_tag.values())
    return {
        "micro": prf(tp, fp, fn),
        "per_tag": {t: prf(c["tp"], c["fp"], c["fn"]) for t, c in sorted(per_tag.items())},
    }
