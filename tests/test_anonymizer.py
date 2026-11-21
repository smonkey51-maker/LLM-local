import unittest

from pii.anonymizer import Anonymizer, iban_valid, luhn_valid, piva_valid, restore


class ChecksumTests(unittest.TestCase):
    def test_iban(self):
        self.assertTrue(iban_valid("IT60 X054 2811 1010 0000 0123 456"))
        self.assertFalse(iban_valid("IT60 X054 2811 1010 0000 0123 457"))

    def test_luhn(self):
        self.assertTrue(luhn_valid("4111 1111 1111 1111"))
        self.assertFalse(luhn_valid("4111 1111 1111 1112"))

    def test_piva(self):
        self.assertTrue(piva_valid("01234567897"))
        self.assertFalse(piva_valid("01234567890"))


class AnonymizerTests(unittest.TestCase):
    def setUp(self):
        self.anon = Anonymizer()

    def test_roundtrip(self):
        src = ("Contatta mario.rossi@studio.it, IBAN IT60 X054 2811 1010 0000 0123 456. "
               "Scrivi ancora a mario.rossi@studio.it.")
        clean, mapping = self.anon.anonymize(src)
        self.assertNotIn("mario.rossi@studio.it", clean)
        self.assertNotIn("IT60", clean)
        # stesso valore -> stesso segnaposto
        self.assertEqual(clean.count("[EMAIL_1]"), 2)
        self.assertEqual(restore(clean, mapping), src)

    def test_invalid_iban_not_masked(self):
        clean, _ = self.anon.anonymize("Codice IT60 X054 2811 1010 0000 0123 457")
        self.assertIn("IT60", clean)

    def test_placeholder_not_confused(self):
        # [EMAIL_1] non deve essere sostituito parzialmente da [EMAIL_10]
        mapping = {f"[EMAIL_{i}]": f"x{i}@a.it" for i in range(1, 12)}
        self.assertEqual(restore("[EMAIL_10] e [EMAIL_1]", mapping), "x10@a.it e x1@a.it")

    def test_custom_detector(self):
        from pii.anonymizer import Entity
        def det(text):
            i = text.find("Rossi")
            return [Entity("FULLNAME", i, i + 5, "Rossi")] if i >= 0 else []
        clean, mapping = Anonymizer([det]).anonymize("Il signor Rossi firma")
        self.assertEqual(clean, "Il signor [FULLNAME_1] firma")
        self.assertEqual(mapping, {"[FULLNAME_1]": "Rossi"})


if __name__ == "__main__":
    unittest.main()
