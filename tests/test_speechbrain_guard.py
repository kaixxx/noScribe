"""Guard against a stale SpeechBrain breaking diarization in a source install.

SpeechBrain is an optional pyannote backend that noScribe does not use, but an
old version left in the environment can fail on import with something other
than ImportError, which pyannote does not catch. ``hide_speechbrain`` hides
it while the pipeline is loaded.

The guard used to wrap only ``from pyannote.audio import Pipeline``. That
import does not load the module with the optional SpeechBrain import --
pyannote's package init is lazy -- so the stale package was only reached in
``from_pretrained``, after the guard had already put it back. Both cases import
``Pipeline`` unguarded, as the worker does; the control case proves that the
stub is then reached in ``from_pretrained``, so the guarded case cannot pass
merely because nothing imported SpeechBrain.

Each case runs in a fresh interpreter, since pyannote records whether
SpeechBrain is available once, when the module is first imported.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The bundled pipeline is handed over as a plain path: the worker's
# impres.as_file() over the "pyannote" namespace needs Python 3.12, and CI also
# runs 3.10.
LOAD = """
import sys
from pyannote.audio import Pipeline
from noScribe.pyannote_mp_worker import hide_speechbrain
path = {path!r}
{call}
print("speechbrain left in sys.modules:", "speechbrain" in sys.modules)
print("embedding:", type(pipeline._embedding).__name__)
"""


def _run(tmp_path, call):
    stub = tmp_path / "stub" / "speechbrain"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(
        'raise AttributeError("stale speechbrain")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(stub.parent), str(REPO)])
    return subprocess.run(
        [sys.executable, "-c",
         LOAD.format(path=str(REPO / "pyannote"), call=call)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)


def test_stale_speechbrain_breaks_unguarded_load(tmp_path):
    r = _run(tmp_path, "pipeline = Pipeline.from_pretrained(path)")
    assert r.returncode != 0
    assert "stale speechbrain" in r.stderr


def test_pipeline_loads_with_speechbrain_hidden(tmp_path):
    r = _run(tmp_path, "with hide_speechbrain():\n"
                       "    pipeline = Pipeline.from_pretrained(path)")
    assert r.returncode == 0, r.stderr
    assert "speechbrain left in sys.modules: False" in r.stdout
    assert "embedding: PyannoteAudioPretrainedSpeakerEmbedding" in r.stdout
