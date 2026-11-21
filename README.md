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

## Training su Colab
1. Carica `data/synthetic/train.jsonl` (generato con `python -m data.synthetic.generate --n 5000 --out ...`) su Google Drive.
2. Runtime GPU, poi nella prima cella: `pip install -r requirements-train.txt` (dal repo clonato o caricato).
3. Monta Drive e lancia:
       python -m training.train --data /content/drive/MyDrive/llm-local/train.jsonl --out /content/drive/MyDrive/llm-local/run1
4. Quantizzazione:
       python -m quantize.quantize --model /content/drive/MyDrive/llm-local/run1/final --out /content/drive/MyDrive/llm-local/int8

Solo dati sintetici su Colab. I documenti reali non vanno mai caricati.
