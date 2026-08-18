import unittest
from unittest.mock import patch

import model


class ClubQaMetricTest(unittest.TestCase):
    def test_run_club_qa_uses_language_neutral_token_f1(self):
        dataset = [
            {
                "data": {
                    "paragraphs": [
                        {
                            "context": "Context mock.",
                            "qas": [
                                {"question": "On?", "answers": [{"text": "l'Audiència de Barcelona"}]},
                                {"question": "Que?", "answers": [{"text": "les carreteres"}]},
                            ],
                        }
                    ]
                }
            }
        ]
        answers = iter(["L Audiencia de Barcelona.", "carreteres"])

        class MockModel:
            def generate(self, prompt, max_new_tokens=64):
                return next(answers)

        with patch.object(model, "load_dataset", return_value=dataset):
            result = model.run_club_qa(MockModel(), n_samples=2)

        self.assertEqual(result["token_f1"], 0.8333)


if __name__ == "__main__":
    unittest.main()
