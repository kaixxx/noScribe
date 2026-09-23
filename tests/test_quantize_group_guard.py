"""tools/quantize_voxtral.py must refuse a group size it cannot honour.

Quantisation group sizes are not a free parameter: MLX supports 32, 64 and 128,
and the group has to divide the last dimension of every tensor being quantised.
Neither condition used to be checked, and neither fails loudly on its own.

An unsupported group (say 16) is rejected by MLX -- but only once quantisation
starts, which in this tool is *after* the source weights have been fetched, up
to 48 GB for the 24B model. A supported group that does not divide some tensor
is worse: mlx_lm's quantize_model skips that tensor silently and leaves it
bf16, so the build loads and transcribes and is merely bigger and slower than
documented, with nothing naming the layer that was left out.

The second test pins that upstream behaviour, because the guard's whole design
rests on it: if a future mlx_lm raised instead of skipping, the guard could be
dropped in favour of the exception. That one needs MLX and skips without it;
the argument checks do not, because the tool imports the Apple-Silicon stack
only after them -- which is what lets these run on Linux CI, and what turns a
missing install into a sentence instead of a ModuleNotFoundError traceback.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "quantize_voxtral.py"


def _run(args, cwd):
    """The tool as the user runs it. Every case here must exit during argument
    validation, i.e. before load_voxtral_model touches the network -- a test
    that downloads weights is a test nobody runs."""
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=cwd, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("group", ["16", "100"])
def test_an_unsupported_group_is_refused_before_the_download(group, tmp_path):
    r = _run(["mistralai/Voxtral-Mini-3B-2507", str(tmp_path / "out"), "8", group],
             cwd=REPO)
    assert r.returncode != 0
    msg = r.stdout + r.stderr
    assert "group must be one of" in msg, msg
    assert "loading" not in msg, "validation ran after the model load started"


@pytest.mark.parametrize("group", ["64"])
def test_a_supported_group_gets_past_the_group_check(group, tmp_path):
    """Positive control: without it the test above would also pass if the tool
    rejected every group. An existing output dir is the next guard in line, so
    reaching *that* message proves the group was accepted and proves it without
    a download."""
    out = tmp_path / "out"
    out.mkdir()
    r = _run(["mistralai/Voxtral-Mini-3B-2507", str(out), "8", group], cwd=REPO)
    assert r.returncode != 0
    msg = r.stdout + r.stderr
    assert "already exists" in msg, msg
    assert "group must be one of" not in msg


def test_mlx_lm_skips_an_indivisible_tensor_silently():
    """The premise of the guard, pinned against the pinned mlx_lm.

    quantize_model wraps any custom predicate in one that returns False when
    `weight.shape[-1] % group_size != 0`, using the *global* group size and
    running before the custom predicate. So the tensor is not quantised, no
    exception is raised, and mlx_voxtral's own fallback to group_size 32 never
    gets the chance to rescue it -- which is why the tool refuses up front
    rather than adopting that fallback.
    """
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.utils import quantize_model
    voxtral_pred = pytest.importorskip(
        "mlx_voxtral.quantization").voxtral_mixed_quantization_predicate

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.good = nn.Linear(128, 8, bias=False)   # 128 % 64 == 0
            self.odd = nn.Linear(96, 8, bias=False)     # 96 % 64 != 0

    for name, pred in (
        ("ours", lambda p, m, *r: {"group_size": 64, "bits": 8}),
        # the library's own predicate asks for group_size 32 on `odd`
        ("library", lambda p, m, *r: voxtral_pred(p, m, {}, 8)),
    ):
        model, cfg = quantize_model(_Model(), {}, group_size=64, bits=8,
                                    quant_predicate=pred)
        kinds = {n: type(m).__name__ for n, m in model.named_modules() if n}
        assert kinds["good"] == "QuantizedLinear", (name, kinds)
        assert kinds["odd"] == "Linear", (
            f"{name}: mlx_lm no longer skips an indivisible tensor -- if it "
            f"raises now, the guard in tools/quantize_voxtral.py can go")
        assert "odd" not in cfg["quantization"], (
            "a skipped tensor must not appear in the quantization map")
