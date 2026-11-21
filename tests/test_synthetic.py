import random
import unittest

from data.synthetic.generate import (
    GENERATORS, cf_check_char, generate, make_card, make_cf, make_iban, make_piva,
)
from pii.anonymizer import iban_valid, luhn_valid, piva_valid, regex_entities


class IdentifierTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0)

    def test_cf_checksum(self):
        for _ in range(500):
            cf = make_cf(self.rng)
            self.assertEqual(cf_check_char(cf[:15]), cf[15], cf)

    def test_cf_known_example(self):
        # esempio di CF con checksum corretto, usato come riferimento
        self.assertEqual(cf_check_char("RSSMRA85T10A562"), "S")

    def test_piva_checksum(self):
        for _ in range(500):
            self.assertTrue(piva_valid(make_piva(self.rng)))

    def test_iban_checksum(self):
        for _ in range(500):
            self.assertTrue(iban_valid(make_iban(self.rng)))

    def test_card_luhn(self):
        for _ in range(500):
            self.assertTrue(luhn_valid(make_card(self.rng)))


class SpanTests(unittest.TestCase):
    def test_spans_match_text(self):
        for ex in generate(300, seed=1):
            for sp in ex.spans:
                self.assertTrue(sp["end"] <= len(ex.text))
                self.assertTrue(ex.text[sp["start"]:sp["end"]])

    def test_spans_do_not_overlap(self):
        for ex in generate(300, seed=2):
            ordered = sorted(ex.spans, key=lambda s: s["start"])
            for a, b in zip(ordered, ordered[1:]):
                self.assertLessEqual(a["end"], b["start"], ex.text)

    def test_all_tags_known(self):
        for ex in generate(100, seed=3):
            for sp in ex.spans:
                self.assertIn(sp["tag"], GENERATORS)

    def test_regex_recovers_structured_spans(self):
        # le entità strutturate devono essere rilevate dal livello regex
        structured = {"CF", "PIVA", "IBAN", "CARD"}
        hits, total = 0, 0
        for ex in generate(300, seed=4):
            found = {(e.start, e.end) for e in regex_entities(ex.text)}
            for sp in ex.spans:
                if sp["tag"] in structured:
                    total += 1
                    hits += (sp["start"], sp["end"]) in found
        self.assertGreater(total, 0)
        self.assertGreater(hits / total, 0.95, f"{hits}/{total}")


if __name__ == "__main__":
    unittest.main()
