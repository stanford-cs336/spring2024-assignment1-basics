#!/usr/bin/env python3
import numpy
import torch
import torch.nn.functional as F

from .adapters import (
    run_gelu,
    run_multihead_self_attention,
    run_positionwise_feedforward,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
)
from .common import FIXTURES_PATH


def test_positionwise_feedforward():
    reference_weights = torch.load(
        FIXTURES_PATH / "positionwise_feedforward_weights.pt"
    )
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(
        FIXTURES_PATH / "positionwise_feedforward_expected_output.pt"
    )
    d_model = 64
    d_ff = 128

    actual_output = run_positionwise_feedforward(
        d_model=d_model, d_ff=d_ff, weights=reference_weights, in_features=in_features
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_scaled_dot_product_attention():
    torch.manual_seed(42)
    # Take the first batch item, so we test the 3D case
    # (input shape (batch_size, seq_len, d_k)) for scaled dot-product attention.
    K = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_K.pt")[0]
    Q = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_Q.pt")[0]
    V = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_V.pt")[0]
    mask = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_mask.pt")
    pdrop = 0.0
    expected_output = torch.load(
        FIXTURES_PATH / "scaled_dot_product_attention_expected_output.pt"
    )[0]
    actual_output = run_scaled_dot_product_attention(
        K=K, Q=Q, V=V, mask=mask, pdrop=pdrop
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_4d_scaled_dot_product_attention():
    torch.manual_seed(42)
    # Shape: (batch_size, num_heads, seq_len, d_k)
    K = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_K.pt")
    Q = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_Q.pt")
    V = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_V.pt")
    mask = torch.load(FIXTURES_PATH / "scaled_dot_product_attention_mask.pt")
    pdrop = 0.0
    expected_output = torch.load(
        FIXTURES_PATH / "scaled_dot_product_attention_expected_output.pt"
    )
    actual_output = run_scaled_dot_product_attention(
        K=K, Q=Q, V=V, mask=mask, pdrop=pdrop
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_multihead_self_attention():
    reference_weights = torch.load(
        FIXTURES_PATH / "unbatched_multihead_self_attention_weights.pt"
    )
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(
        FIXTURES_PATH / "unbatched_multihead_self_attention_expected_output.pt"
    )
    d_model = 64
    num_heads = 2
    attn_pdrop = 0.0
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        attn_pdrop=attn_pdrop,
        weights=reference_weights,
        in_features=in_features,
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_transformer_lm():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    in_indices = torch.load(FIXTURES_PATH / "in_indices.pt")
    expected_output = torch.load(FIXTURES_PATH / "transformer_lm_expected_output.pt")
    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices,
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-4
    )


def test_transformer_lm_truncated_input():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    in_indices_truncated = torch.load(FIXTURES_PATH / "in_indices_truncated.pt")
    truncated_expected_output = torch.load(
        FIXTURES_PATH / "transformer_lm_truncated_expected_output.pt"
    )
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices_truncated,
    )
    numpy.testing.assert_allclose(
        truncated_actual_output.detach().numpy(),
        truncated_expected_output.detach().numpy(),
        atol=1e-4,
    )


def test_transformer_block():
    torch.manual_seed(42)
    reference_weights = torch.load(FIXTURES_PATH / "transformer_block_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(FIXTURES_PATH / "transformer_block_expected_output.pt")
    d_model = 64
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_features=in_features,
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_rmsnorm():
    reference_weights = torch.load(FIXTURES_PATH / "rmsnorm_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(FIXTURES_PATH / "rmsnorm_expected_output.pt")
    d_model = 64
    actual_output = run_rmsnorm(
        d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_features
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_gelu():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = torch.tensor(
        [
            [
                0.13946731388568878,
                0.7617851495742798,
                0.3622361421585083,
                0.3221103549003601,
                0.8121858239173889,
            ],
            [
                0.5881373286247253,
                0.8080969452857971,
                0.1243969276547432,
                0.7709409594535828,
                0.007538566831499338,
            ],
        ]
    )
    actual_output = run_gelu(x)
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )


def test_gelu_matches_pytorch():
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )
    expected_output = F.gelu(x)
    actual_output = run_gelu(x)
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )
