"""Anonimizzazione locale reversibile di testi (MVP ispirato a rizzo-pii).

Pipeline:
1. regex + validatori (IBAN mod-97, carte Luhn, P.IVA, email, URL, CF, telefono)
2. detector opzionali (es. modello HF token-classification), con priorità inferiore
3. segnaposto numerati stabili ([IBAN_1]) + mappa locale per il ripristino
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Entity:
    tag: str
    start: int
    end: int
    text: str


# ---------- validatori ----------

def iban_valid(value: str) -> bool:
    s = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(c, 36)) for c in rearranged)
    return int(numeric) % 97 == 1


def luhn_valid(value: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def piva_valid(value: str) -> bool:
    if not re.fullmatch(r"\d{11}", value):
        return False
    total = 0
    for i, ch in enumerate(value[:10]):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10 == int(value[10])


# ---------- regole regex ----------
# (tag, pattern, flags, validatore opzionale)
RULES: list[tuple[str, str, int, Callable[[str], bool] | None]] = [
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", 0, None),
    ("URL", r"https?://\S+|www\.\S+", 0, None),
    ("IBAN", r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b", 0, iban_valid),
    ("CARD", r"\b\d(?:[ -]?\d){12,18}\b", 0, luhn_valid),
    ("CF", r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b", re.IGNORECASE, None),
    ("PIVA", r"\b\d{11}\b", 0, piva_valid),
    ("PHONE", r"(?<!\w)(?:\+39[ .-]?)?3\d{2}[ .-]?\d{6,7}\b"
              r"|(?<!\w)0\d{1,3}[ .-]?\d{5,8}\b", 0, None),
]


def regex_entities(text: str) -> list[Entity]:
    found: list[Entity] = []
    for tag, pattern, flags, validator in RULES:
        for m in re.finditer(pattern, text, flags):
            value = m.group(0).rstrip()
            if validator and not validator(value):
                continue
            found.append(Entity(tag, m.start(), m.start() + len(value), value))
    return found


# ---------- detector modello (opzionale) ----------

Detector = Callable[[str], Iterable[Entity]]


def hf_detector(model_id: str) -> Detector:
    """Wrappa una pipeline HF token-classification. Non testato con rizzo-pii:
    i nomi delle label del modello vanno verificati sul model card."""
    from transformers import pipeline  # import pigro: dipendenza opzionale

    nlp = pipeline("token-classification", model=model_id,
                   aggregation_strategy="simple")

    def run(text: str) -> list[Entity]:
        out = []
        for ent in nlp(text):
            out.append(Entity(str(ent["entity_group"]).upper(),
                              int(ent["start"]), int(ent["end"]),
                              text[ent["start"]:ent["end"]]))
        return out

    return run


def _overlaps(a: Entity, b: Entity) -> bool:
    return a.start < b.end and b.start < a.end


# ---------- anonimizzatore ----------

PLACEHOLDER_RE = re.compile(r"\[[A-Z_]+\d+\]")


class Anonymizer:
    def __init__(self, detectors: list[Detector] | None = None):
        self.detectors = detectors or []

    def detect(self, text: str) -> list[Entity]:
        accepted = regex_entities(text)  # le regole hanno priorità
        for det in self.detectors:
            for ent in det(text):
                if not any(_overlaps(ent, a) for a in accepted):
                    accepted.append(ent)
        return sorted(accepted, key=lambda e: e.start)

    def anonymize(self, text: str) -> tuple[str, dict[str, str]]:
        """Ritorna (testo_anonimizzato, mappa segnaposto -> valore reale)."""
        counters: dict[str, int] = {}
        reverse: dict[tuple[str, str], str] = {}
        mapping: dict[str, str] = {}
        out: list[str] = []
        pos = 0
        for ent in self.detect(text):
            out.append(text[pos:ent.start])
            key = (ent.tag, ent.text)
            if key not in reverse:  # stesso valore -> stesso segnaposto
                counters[ent.tag] = counters.get(ent.tag, 0) + 1
                ph = f"[{ent.tag}_{counters[ent.tag]}]"
                reverse[key] = ph
                mapping[ph] = ent.text
            out.append(reverse[key])
            pos = ent.end
        out.append(text[pos:])
        return "".join(out), mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    """Sostituisce i segnaposto nella risposta del modello con i valori reali."""
    return PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
