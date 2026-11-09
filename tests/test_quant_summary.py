"""Tests for _quant_summary: the log line must state the *exact* build.

The config's quantization dict only lists what was quantised; deliberately
dense components (this project's bf16 audio encoder) appear nowhere in it.
The summary therefore also checks the weight names for components without
quantisation scales -- these tests pin that down with synthetic model dirs.
"""
import json
import struct

import pytest

from noScribe.voxtral_engine import _quant_summary


def _write_model(tmp_path, config, weight_names=None, sharded=True):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if weight_names is None:
        return tmp_path
    if sharded:
        idx = {"weight_map": {n: "model-00001.safetensors" for n in weight_names}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(idx))
    else:
        header = json.dumps({n: {} for n in weight_names}).encode()
        (tmp_path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header)
    return tmp_path


MINI_8BIT_CFG = {
    "torch_dtype": "bfloat16",
    "quantization": {
        "bits": 8, "group_size": 64, "mode": "affine",
        "lm_head": {"group_size": 64, "bits": 8},
        "language_model.layers.0.mlp.down_proj": {"group_size": 64, "bits": 8},
    },
}
MINI_8BIT_WEIGHTS = [
    "audio_tower.conv1.weight",
    "multi_modal_projector.linear_1.weight",
    "language_model.layers.0.mlp.down_proj.weight",
    "language_model.layers.0.mlp.down_proj.scales",
    "lm_head.weight", "lm_head.scales",
    "embed_tokens.weight", "embed_tokens.scales",
]


def test_dense_components_are_reported(tmp_path):
    """The deliberate asymmetry of this project's builds -- bf16 audio encoder
    and projector, quantised LM -- must be visible in the summary."""
    repo = _write_model(tmp_path, MINI_8BIT_CFG, MINI_8BIT_WEIGHTS)
    assert _quant_summary(repo) == \
        "8-bit, group 64, affine; audio encoder & projector bf16"


def test_single_file_safetensors_header_is_read(tmp_path):
    repo = _write_model(tmp_path, MINI_8BIT_CFG, MINI_8BIT_WEIGHTS, sharded=False)
    assert "audio encoder & projector bf16" in _quant_summary(repo)


def test_mixed_bit_widths_are_listed(tmp_path):
    cfg = {
        "torch_dtype": "bfloat16",
        "quantization": {
            "bits": 4, "group_size": 64, "mode": "affine",
            "lm_head": {"group_size": 64, "bits": 8},
            "embed_tokens": False,
        },
    }
    repo = _write_model(tmp_path, cfg)  # no weights file: config-only report
    assert _quant_summary(repo) == \
        "4-bit, group 64, affine; embed_tokens bf16, lm_head 8-bit"


def test_unquantised_build_reports_dtype(tmp_path):
    repo = _write_model(tmp_path, {"torch_dtype": "bfloat16"})
    assert _quant_summary(repo) == "bf16"


def test_unreadable_repo_is_silent():
    assert _quant_summary("no/such/dir") == ""
