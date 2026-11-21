import unittest

from training.bio import IGNORE, build_label_maps, decode_chars, decode_tokens, evaluate, token_labels

L2I, I2L = build_label_maps(["CF", "IBAN"])


class TokenLabelTests(unittest.TestCase):
    def test_bio_on_two_tokens_span(self):
        # testo: "CF ABC DEF" ; span su "ABC DEF" (caratteri 3-10)
        offsets = [(0, 0), (0, 2), (3, 6), (7, 10), (0, 0)]
        spans = [{"start": 3, "end": 10, "tag": "CF"}]
        labels = token_labels(offsets, spans, L2I)
        self.assertEqual(labels, [IGNORE, L2I["O"], L2I["B-CF"], L2I["I-CF"], IGNORE])

    def test_special_tokens_ignored(self):
        labels = token_labels([(0, 0), (0, 3)], [], L2I)
        self.assertEqual(labels, [IGNORE, L2I["O"]])


class DecodeTests(unittest.TestCase):
    def test_decode_tokens_chunks(self):
        ids = [L2I["B-CF"], L2I["I-CF"], L2I["O"], L2I["B-IBAN"]]
        self.assertEqual(decode_tokens(ids, I2L), {("CF", 0, 1), ("IBAN", 3, 3)})

    def test_adjacent_different_tags_split(self):
        ids = [L2I["B-CF"], L2I["B-IBAN"]]
        self.assertEqual(decode_tokens(ids, I2L), {("CF", 0, 0), ("IBAN", 1, 1)})

    def test_decode_chars_uses_offsets(self):
        offsets = [(0, 0), (3, 6), (7, 10)]
        ids = [IGNORE, L2I["B-CF"], L2I["I-CF"]]
        self.assertEqual(decode_chars(offsets, ids, I2L), [{"tag": "CF", "start": 3, "end": 10}])

    def test_roundtrip_gold_labels(self):
        offsets = [(0, 0), (0, 2), (3, 6), (7, 10), (11, 14), (0, 0)]
        spans = [{"start": 3, "end": 10, "tag": "CF"}, {"start": 11, "end": 14, "tag": "IBAN"}]
        labels = token_labels(offsets, spans, L2I)
        chars = decode_chars(offsets, labels, I2L)
        self.assertEqual(chars, spans)


class EvaluateTests(unittest.TestCase):
    def test_perfect(self):
        gold = [{("CF", 0, 1)}]
        res = evaluate(gold, gold)
        self.assertEqual(res["micro"]["f1"], 1.0)

    def test_partial(self):
        gold = [{("CF", 0, 1), ("IBAN", 3, 3)}]
        pred = [{("CF", 0, 1), ("CF", 5, 5)}]
        res = evaluate(gold, pred)
        self.assertEqual(res["micro"]["precision"], 0.5)
        self.assertEqual(res["micro"]["recall"], 0.5)
        self.assertEqual(res["per_tag"]["IBAN"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
