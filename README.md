# llm-local

Anonimizzazione locale di documenti italiani (PDF, Word, testo) prima dell'invio a LLM cloud.
Il modello di rilevazione gira in locale; la mappa segnaposto → valore reale resta sul proprio computer.

## Stato
Scaffolding iniziale. Il modulo `pii/` contiene il primo prototipo (regex + checksum).

## Struttura
- `app/` API HTTP locale
- `pii/` rilevazione, segnaposto e ripristino
- `data/synthetic/` generatori di dati sintetici (versionati)
- `data/real/` documenti reali: esclusi da git
- `training/` fine-tuning del modello
- `quantize/` notebook per la quantizzazione (Colab)
- `eval/` valutazione per span, falsi positivi, set di test
- `tests/` test automatici

## Avvio
    pip install -r requirements.txt
    python -m app.app
    python -m unittest discover -s tests -t .
