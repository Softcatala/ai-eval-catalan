import unittest
from unittest.mock import patch

import model


class CatcolaParserTest(unittest.TestCase):
    def test_parse_catcola_answer_accepts_only_prefix_si_or_no(self):
        cases = {
            "si": 1,
            "sí": 1,
            "Si.": 1,
            "Sí, és correcta.": 1,
            "no": 0,
            "No.": 0,
            "No ho és.": 0,
            "": None,
            "correcte": None,
            "Resposta: sí": None,
            "potser no": None,
        }

        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(model.parse_catcola_answer(answer), expected)

    def test_run_catcola_counts_invalid_as_strict_accuracy_failures(self):
        dataset = [
            {"Sentence": "Frase acceptable.", "Label": 1},
            {"Sentence": "Frase no acceptable.", "Label": 0},
            {"Sentence": "Resposta buida.", "Label": 1},
            {"Sentence": "Resposta fora de format.", "Label": 0},
        ]

        class MockModel:
            def __init__(self):
                self.answers = iter(["sí", "no", "", "correcte"])

            def generate(self, prompt, max_new_tokens=16):
                return next(self.answers)

        with patch.object(model, "load_dataset", return_value=dataset):
            result = model.run_catcola(MockModel(), n_samples=4)

        self.assertEqual(result["mcc"], 1.0)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid_rate"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["n_valid"], 2)
        self.assertEqual(result["n_invalid"], 2)


if __name__ == "__main__":
    unittest.main()
