"""Exercise real torch model inference with tiny local fixtures, no model download."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from guardlens.config import GuardLensConfig
from guardlens.models.guardlens import GuardLens
from eval_platform.runtime import GuardLensBackend, CoverageError, ShieldGemmaBackend, LLMAttributor
from test_eval_platform import record


class TinyTokenizer:
    is_fast = True
    def __call__(self, texts, **kwargs):
        if kwargs.get("truncation") is not False:
            raise AssertionError("must never truncate")
        rows = [[1] + [3 + ord(c) % 40 for c in text] + [2] for text in texts]
        offsets = [[(0, 0)] + [(i, i + 1) for i in range(len(text))] + [(0, 0)] for text in texts]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]),
                "offset_mapping": torch.tensor([r + [(0, 0)] * (width - len(r)) for r in offsets])}


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50, 12)
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def setup(model):
    model.backbone = TinyBackbone()
    model.backbone_loaded = True


class RuntimeTests(unittest.TestCase):
    def test_current_checkpoint_load_and_offsets(self):
        config = GuardLensConfig(backbone_dim=12, cross_turn_dim=8, cross_turn_heads=2, cross_turn_layers=1,
            attr_hidden_dim=6, cls_hidden_dim=8, max_turns=4, max_tokens_per_turn=40,
            turn_pooling="mean")
        model = GuardLens(config)
        setup(model)
        ckpt = {"architecture_version": "causal_localization_v2", "training_contract_version": "restored_a_primary_plus_auxiliary_v2",
            "model_name": "guardlens", "phase": 2, "config": config, "threshold": .63, "model_state_dict": model.state_dict(),
            "data_sha256": {"train": "a", "dev": "b"}, "code_sha": "c", "epoch": 10, "score_name": "joint"}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "best_joint.pt"
            torch.save(ckpt, path)
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), patch.object(GuardLens, "setup_backbone", setup):
                backend = GuardLensBackend(path, "cpu")
            turns = record()["turns"]
            prediction = backend.predict(turns)
            self.assertEqual(set(prediction["turn_scores"]), {"0", "2"})
            self.assertEqual({s["turn_id"] for s in prediction["token_scores"]}, {0, 2})
            self.assertTrue(all(s["end"] > s["start"] for s in prediction["token_scores"]))
            self.assertEqual(backend.threshold, .63)
            self.assertEqual(prediction, backend.predict(turns))
            with self.assertRaises(CoverageError):
                backend.predict([{**turns[0], "text": "x" * 50}])
            with self.assertRaises(CoverageError):
                backend.predict([{**turns[0], "role": "system"}])
            ckpt["model_name"] = "unregistered"
            torch.save(ckpt, path)
            with self.assertRaises(ValueError):
                GuardLensBackend(path, "cpu")

    def test_shield_scores_four_policies_and_preserves_early_context(self):
        class Tokenizer:
            def get_vocab(self):
                return {"Yes": 0, "No": 1}
            def decode(self, ids):
                if isinstance(ids, list):
                    return "Yes" if ids == [0] else "No"
                return "prompt"
        class Chat:
            identity = {"model": "fixture"}
            tokenizer = Tokenizer()
            def __init__(self):
                self.calls = []
                self.model_calls = []
                self.torch = torch
                self.model = self.forward
            def forward(self, *, input_ids, use_cache, logits_to_keep):
                self.model_calls.append({"use_cache": use_cache, "logits_to_keep": logits_to_keep})
                return SimpleNamespace(logits=torch.tensor([[[2., 0.]]]))
            def encode(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return {"input_ids": torch.tensor([[2]])}
        chat = Chat()
        policies = {k: k for k in ("dangerous", "harassment", "hate", "sexual")}
        backend = ShieldGemmaBackend(chat, policies)
        output = backend.predict(record()["turns"])
        self.assertEqual(len(chat.calls), 4)
        self.assertEqual(chat.model_calls, [{"use_cache": False, "logits_to_keep": 1}] * 4)
        self.assertEqual(set(output["category_scores"]), set(policies))
        self.assertAlmostEqual(output["probability"], .880797, places=5)
        self.assertIn("alpha", chat.calls[0][0][0]["content"])
        self.assertIn("assistant", chat.calls[0][0][0]["content"])

    def test_llm_parser_rejects_missing_turns_and_bad_offsets(self):
        class Chat:
            identity = {"model": "fixture"}
            def __init__(self, value):
                self.value = value
            def generate(self, messages, seed):
                return {"text": json.dumps(self.value), "hit_output_limit": False}
        value = {"probability": .8, "turns": {"0": .8}, "spans": []}
        self.assertIn("error", LLMAttributor(Chat(value)).predict(record()["turns"]))
        value["turns"]["2"] = .1
        value["spans"] = [{"turn_id": 0, "start": 0, "end": 999, "score": .8}]
        self.assertIn("error", LLMAttributor(Chat(value)).predict(record()["turns"]))
        value["spans"] = []
        self.assertNotIn("error", LLMAttributor(Chat(value)).predict(record()["turns"]))


if __name__ == "__main__":
    unittest.main()
