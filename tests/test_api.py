import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import api


class FakeModel:
    embedding = type("Embedding", (), {"num_embeddings": 32})()

    def __init__(self, predicted_index):
        self.predicted_index = predicted_index
        self.calls = 0

    def __call__(self, inputs):
        self.calls += 1
        logits = torch.zeros((1, 3))
        logits[0, self.predicted_index] = 10
        attention = torch.ones((1, inputs.shape[1])) / inputs.shape[1]
        return logits, attention


class AnalyzeEndpointTests(unittest.TestCase):
    def test_functional_error_preempts_neural_inference(self):
        model = FakeModel(predicted_index=0)
        with patch.object(api, "model", model), patch.object(
            api, "tokenizer", {"<UNK>": 0}
        ):
            result = api.analyze(api.CodeRequest(code='value = 1 + "text"'))

        self.assertEqual(result["category"], "Error")
        self.assertEqual(result["issues"][0]["priority"], "high")
        self.assertEqual(model.calls, 0)
        self.assertEqual(result["attention_map"], [])

    def test_syntax_error_uses_public_error_category(self):
        model = FakeModel(predicted_index=0)
        with patch.object(api, "model", model):
            result = api.analyze(api.CodeRequest(code="if True print(1)"))

        self.assertEqual(result["category"], "Error")
        self.assertEqual(result["issues"][0]["code"], "SyntaxError")
        self.assertEqual(result["issues"][0]["priority"], "high")
        self.assertEqual(model.calls, 0)
        self.assertEqual(result["attention_map"], [])

    def test_legacy_neural_category_is_not_reported_as_functional_error(self):
        model = FakeModel(predicted_index=2)
        with patch.object(api, "model", model), patch.object(
            api, "tokenizer", {"<UNK>": 0}
        ):
            result = api.analyze(api.CodeRequest(code="print('ok')"))

        self.assertEqual(result["category"], "Clean")
        self.assertEqual(result["issues"][0]["code"], "Clean")
        self.assertTrue(result["attention_breakdown"])
        self.assertEqual(model.calls, 1)
        attention_map = result["attention_map"]
        self.assertEqual(
            [item["token"] for item in attention_map],
            result["ast_sequence"].split(),
        )
        self.assertAlmostEqual(sum(item["weight"] for item in attention_map), 1.0)
        self.assertTrue(all(0 <= item["weight"] <= 1 for item in attention_map))


if __name__ == "__main__":
    unittest.main()
