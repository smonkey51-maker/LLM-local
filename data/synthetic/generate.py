"""Generatore di esempi sintetici in italiano con span annotati.

Output JSONL: {"text": str, "spans": [{"start", "end", "tag"}]}
Gli identificativi strutturati (CF, P.IVA, IBAN, carte) rispettano i checksum reali.
Nessun dato reale: tutti i valori sono casuali.

    python -m data.synthetic.generate --n 5000 --seed 42 --out data/synthetic/train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass

# ---------- generatori di identificativi con checksum ----------

_CF_ODD = dict(zip("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", [
    1, 0, 5, 7, 9, 13, 15, 17, 19, 21,          # cifre 0-9
    1, 0, 5, 7, 9, 13, 15, 17, 19, 21, 2, 4, 18, 20, 11, 3, 6, 8, 12, 14, 16, 10, 22, 25, 24, 23]))
_CF_EVEN = {c: i for i, c in enumerate("0123456789")}
_CF_EVEN.update({c: i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")})


def cf_check_char(first15: str) -> str:
    total = 0
    for i, ch in enumerate(first15):
        total += _CF_ODD[ch] if i % 2 == 0 else _CF_EVEN[ch]
    return chr(ord("A") + total % 26)


def make_cf(rng: random.Random) -> str:
    letters = "ABCDEHLMPRST"
    body = (
        "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(6))
        + f"{rng.randint(0, 99):02d}"
        + rng.choice(letters)
        + f"{rng.randint(1, 70):02d}"
        + rng.choice("ABCDEFGHIL")
        + f"{rng.randint(0, 999):03d}"
    )
    return body + cf_check_char(body)


def piva_check_digit(first10: str) -> int:
    total = 0
    for i, ch in enumerate(first10):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def make_piva(rng: random.Random) -> str:
    first10 = "".join(rng.choice("0123456789") for _ in range(10))
    return first10 + str(piva_check_digit(first10))


def make_iban(rng: random.Random) -> str:
    cin = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    abi = f"{rng.randint(1, 99999):05d}"
    cab = f"{rng.randint(1, 99999):05d}"
    account = "".join(rng.choice("0123456789") for _ in range(12))
    bban = cin + abi + cab + account
    numeric = "".join(str(int(c, 36)) for c in bban + "IT00")
    check = 98 - int(numeric) % 97
    return f"IT{check:02d}{bban}"


def _luhn_check_digit(digits: str) -> int:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d)
        if i % 2 == 0:  # posizioni che verranno raddoppiate dopo l'aggiunta del check
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return (10 - total % 10) % 10


def make_card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "51", "52", "53", "54", "55"])
    body = prefix + "".join(rng.choice("0123456789") for _ in range(15 - len(prefix)))
    number = body + str(_luhn_check_digit(body))
    groups = [number[i:i + 4] for i in range(0, 16, 4)]
    return " ".join(groups)


# ---------- valori generici ----------

FIRST_NAMES = ["Marco", "Giulia", "Luca", "Francesca", "Andrea", "Chiara", "Matteo",
               "Sara", "Alessandro", "Valentina", "Davide", "Elena", "Stefano", "Laura"]
LAST_NAMES = ["Rossi", "Bianchi", "Ferrari", "Esposito", "Romano", "Colombo", "Ricci",
              "Marino", "Greco", "Bruno", "Gallo", "Conti", "De Luca", "Costa", "Forcolin"]
STREETS = ["Via Roma", "Corso Italia", "Via Garibaldi", "Piazza Duomo", "Via Verdi",
           "Via dei Mille", "Viale della Libertà", "Via Dante"]
CITIES = [("Padova", "35100", "PD"), ("Milano", "20121", "MI"), ("Roma", "00184", "RM"),
          ("Torino", "10121", "TO"), ("Bologna", "40121", "BO"), ("Napoli", "80133", "NA")]
DOMAINS = ["gmail.com", "libero.it", "outlook.it", "studiolegale.it", "azienda.it"]
PEC_DOMAINS = ["pec.it", "legalmail.it", "postacert.it"]
URLS = ["https://www.example-ente.it/pratiche", "https://portale.example.gov.it/istanze"]


def make_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def make_email(rng: random.Random, name: str | None = None) -> str:
    base = (name or make_name(rng)).lower().replace(" ", ".")
    return f"{base}@{rng.choice(DOMAINS)}"


def make_pec(rng: random.Random) -> str:
    user = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(6, 12)))
    return f"{user}@{rng.choice(PEC_DOMAINS)}"


def make_phone(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return f"3{rng.randint(10, 99)} {rng.randint(1000000, 9999999)}"
    return f"+39 3{rng.randint(10, 99)} {rng.randint(100000, 9999999)}"


def make_address(rng: random.Random) -> str:
    city, cap, prov = rng.choice(CITIES)
    return f"{rng.choice(STREETS)} {rng.randint(1, 180)}, {cap} {city} ({prov})"


def make_pratica(rng: random.Random) -> str:
    return f"PR-{rng.randint(2020, 2026)}/{rng.randint(1, 99999):05d}"


def make_targa(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPRSTVXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}{rng.randint(0, 999):03d}{rng.choice(letters)}{rng.choice(letters)}"


def make_catasto(rng: random.Random) -> str:
    return f"foglio {rng.randint(1, 150)} particella {rng.randint(1, 2000)}"


def make_url(rng: random.Random) -> str:
    return f"{rng.choice(URLS)}/{rng.randint(1000, 9999)}"


GENERATORS = {
    "FULLNAME": make_name,
    "CF": make_cf,
    "PIVA": make_piva,
    "IBAN": make_iban,
    "CARD": make_card,
    "EMAIL": make_email,
    "PEC": make_pec,
    "PHONE": make_phone,
    "ADDRESS": make_address,
    "PRATICA": make_pratica,
    "TARGA": make_targa,
    "CATASTO": make_catasto,
    "URL": make_url,
}

# ---------- template di frasi ----------
# {TAG} viene sostituito da un valore generato; il resto è testo fisso.
TEMPLATES = [
    "Il sig. {FULLNAME}, codice fiscale {CF}, residente in {ADDRESS}, ha presentato la pratica {PRATICA}.",
    "La società con partita IVA {PIVA} chiede il rimborso sull'IBAN {IBAN} intestato a {FULLNAME}.",
    "Inviare le comunicazioni alla PEC {PEC} oppure a {EMAIL}, telefono {PHONE}.",
    "Il veicolo targato {TARGA} risulta intestato a {FULLNAME} dal 2019.",
    "L'immobile è censito al catasto urbano, {CATASTO}, sito in {ADDRESS}.",
    "Il pagamento di 350,00 euro è stato addebitato sulla carta {CARD} il giorno 12 marzo.",
    "Per informazioni scrivere a {FULLNAME} all'indirizzo {EMAIL}.",
    "Consultare la documentazione su {URL} allegando la pratica {PRATICA}.",
    "Il contratto è stato firmato dal legale rappresentante, nato a {ADDRESS}, C.F. {CF}.",
    "Ai sensi dell'art. 13 del GDPR, il titolare tratta i dati di {FULLNAME} per la finalità indicata.",
    # frasi senza PII: servono a limitare i falsi positivi
    "La riunione è fissata per il 14 novembre alle ore 10 nella sala riunioni del primo piano.",
    "L'importo complessivo dell'appalto ammonta a 120.000 euro, IVA esclusa.",
]

_SLOT = re.compile(r"\{([A-Z]+)\}")


@dataclass
class Example:
    text: str
    spans: list[dict]

    def to_json(self) -> str:
        return json.dumps({"text": self.text, "spans": self.spans}, ensure_ascii=False)


def render(template: str, rng: random.Random) -> Example:
    parts: list[str] = []
    spans: list[dict] = []
    pos = 0
    length = 0
    for m in _SLOT.finditer(template):
        literal = template[pos:m.start()]
        parts.append(literal)
        length += len(literal)
        tag = m.group(1)
        value = GENERATORS[tag](rng)
        spans.append({"start": length, "end": length + len(value), "tag": tag})
        parts.append(value)
        length += len(value)
        pos = m.end()
    parts.append(template[pos:])
    return Example("".join(parts), spans)


def generate(n: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    return [render(rng.choice(TEMPLATES), rng) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    examples = generate(args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")
    print(f"scritti {len(examples)} esempi in {args.out}")


if __name__ == "__main__":
    main()
