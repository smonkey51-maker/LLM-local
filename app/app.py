"""API HTTP locale. Ascolta solo su 127.0.0.1.

Avvio:
    python app.py                       # solo regex
    RIZZO_MODEL=<id_hf> python app.py   # regex + modello HF
"""
import os

from flask import Flask, jsonify, request

from pii.anonymizer import Anonymizer, hf_detector, restore

detectors = []
if os.environ.get("RIZZO_MODEL"):
    detectors.append(hf_detector(os.environ["RIZZO_MODEL"]))

anon = Anonymizer(detectors)
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=bool(detectors))


@app.post("/anonymize")
def anonymize():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not isinstance(text, str):
        return jsonify(error="'text' deve essere una stringa"), 400
    clean, mapping = anon.anonymize(text)
    # La mappa contiene dati personali: va tenuta solo in locale.
    return jsonify(text=clean, mapping=mapping)


@app.post("/restore")
def restore_endpoint():
    body = request.get_json(silent=True) or {}
    return jsonify(text=restore(body.get("text", ""), body.get("mapping", {})))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005)
