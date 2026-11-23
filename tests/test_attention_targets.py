"""The qk/qkv --pai-targets: SplitAttention keeps the model's function, and perforated
state_dict keys map back to the plain model (plain_state_dict) and forward again (split_in_proj),
as --pai-resume, write_best_model and export_plain_checkpoint.py need. No PAI needed: a stand-in
for GPA.PAISequential gives the same '<module>.model.0.' key layout."""
import copy
import unittest

import torch
import torch.nn as nn

import pai_setup
from simple_vit import SimpleVisionTransformer


class FakePAISequential(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def tiny_vit():
    torch.manual_seed(0)
    model = SimpleVisionTransformer(image_size=32, patch_size=8, num_layers=3, num_heads=3,
                                    hidden_dim=48, mlp_dim=96, num_classes=10)
    nn.init.normal_(model.heads.head.weight)  # zero-initialized, which would hide any difference
    return model.eval()


def logits(model, x):
    y = torch.zeros(x.shape[0], dtype=torch.long)
    with torch.no_grad():
        return model(x, 1.0, y, y)[0]


class AttentionTargetTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(4, 3, 32, 32)

    def perforated(self, target, layers=(1, 2)):
        plain = tiny_vit()
        model = copy.deepcopy(plain)
        for i in layers:
            pai_setup.wrap_target(model.encoder.layers[i], target, FakePAISequential)
        return plain, model

    def test_split_attention_matches_multihead_attention(self):
        for target in pai_setup.ATTENTION_TARGETS:
            with self.subTest(target=target):
                plain, model = self.perforated(target)
                torch.testing.assert_close(logits(model, self.x), logits(plain, self.x))

    def test_wraps_ln_1_with_the_target_projection(self):
        _, model = self.perforated('qk')
        block = model.encoder.layers[1]
        attention = block.self_attention
        self.assertIsInstance(attention.qk, FakePAISequential)
        self.assertIsInstance(attention.qk.model[0], nn.LayerNorm)
        self.assertEqual(attention.qk.model[1].out_features, 96)
        self.assertIs(attention.ln_1, attention.qk.model[0])  # one LayerNorm for q, k and v
        self.assertIsInstance(attention.v, nn.Linear)
        self.assertIsInstance(block.ln_1, nn.Identity)
        self.assertIsInstance(model.encoder.layers[0].self_attention, nn.MultiheadAttention)
        _, model = self.perforated('qkv')
        self.assertIsInstance(model.encoder.layers[1].self_attention.qkv.model[0], nn.LayerNorm)

    def test_shared_ln_gets_gradients_from_q_k_and_v(self):
        # With the v path on a separate LayerNorm copy, the two copies split the gradient; the
        # shared LayerNorm must receive their sum.
        _, model = self.perforated('qk', layers=(2,))
        model.train()
        y = torch.zeros(self.x.shape[0], dtype=torch.long)
        model(self.x, 1.0, y, y)[1].backward()
        ln = model.encoder.layers[2].self_attention.ln_1
        shared_grad = ln.weight.grad.clone()

        _, untied = self.perforated('qk', layers=(2,))
        attention = untied.encoder.layers[2].self_attention
        attention.ln_1 = copy.deepcopy(attention.ln_1)
        untied.train()
        untied(self.x, 1.0, y, y)[1].backward()
        torch.testing.assert_close(shared_grad, attention.qk.model[0].weight.grad + attention.ln_1.weight.grad)

    def test_check_shared_ln(self):
        _, model = self.perforated('qk')
        pai_setup.check_shared_ln(model)
        attention = model.encoder.layers[2].self_attention
        attention.ln_1 = copy.deepcopy(attention.ln_1)
        with self.assertRaisesRegex(RuntimeError, 'encoder_layer_2.ln_1'):
            pai_setup.check_shared_ln(model)

    def test_plain_state_dict_round_trip(self):
        for target in pai_setup.ATTENTION_TARGETS:
            with self.subTest(target=target):
                plain, model = self.perforated(target)
                state = model.state_dict()
                self.assertEqual(pai_setup.detect_target(state), target)
                restored = tiny_vit()
                for p in restored.parameters():
                    nn.init.zeros_(p)
                restored.load_state_dict(pai_setup.plain_state_dict(state), strict=True)
                for key, value in plain.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value, msg=key)

    def test_split_in_proj_finds_every_perforated_weight(self):
        # write_best_model's lookup: every perforated key -> plain name -> tensor of the same value.
        for target in pai_setup.ATTENTION_TARGETS:
            with self.subTest(target=target):
                plain, model = self.perforated(target)
                best = pai_setup.split_in_proj(plain.state_dict())
                for key, value in model.state_dict().items():
                    name = pai_setup.plain_key(key, target)
                    torch.testing.assert_close(best[name], value, msg=key)


if __name__ == '__main__':
    unittest.main()
