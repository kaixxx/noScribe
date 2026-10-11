"""The Voxtral worker's queue contract, as `main.py` consumes it.

`voxtral_proc_entrypoint` is the spawn target the GUI talks to through one
queue. The contract it must keep (see the worker's docstring): a finished run
ends in `{"type": "result", "ok": True, "info": ...}`; any exception ends in
`ok: False` with the exception's type and message in `error` and the traceback
in `trace`, so the GUI can show one and log the other; `log` and `progress`
puts that fail are swallowed, because losing a log line must not fail a job;
but a `segment` put that fails is deliberately NOT swallowed -- a transcript
silently truncated while the job reports success is worse than a failed job.
"""
import pytest

from noScribe import voxtral_engine, voxtral_mp_worker


class _Queue:
    """Records puts; raises on the message types listed in `broken`."""

    def __init__(self, broken=()):
        self.items = []
        self.broken = set(broken)

    def put(self, msg):
        if msg.get("type") in self.broken:
            raise OSError("queue closed")
        self.items.append(msg)


def _fake_transcribe(segments=({"start": 0.0, "end": 1.0, "text": "hi", "words": None},)):
    def transcribe(**kw):
        kw["log_cb"]("info", "hello")
        kw["progress_cb"](42)
        for seg in segments:
            kw["segment_cb"](seg)
        return list(segments), {"duration": 1.0, "language": "de"}
    return transcribe


def test_ok_run_forwards_log_progress_segments_and_info(monkeypatch):
    monkeypatch.setattr(voxtral_engine, "transcribe", _fake_transcribe())
    q = _Queue()
    voxtral_mp_worker.voxtral_proc_entrypoint({"audio_path": "x.wav"}, q)
    assert [m["type"] for m in q.items] == ["log", "progress", "segment", "result"]
    assert q.items[0] == {"type": "log", "level": "info", "msg": "hello"}
    assert q.items[1] == {"type": "progress", "pct": 42}
    assert q.items[2]["segment"]["text"] == "hi"
    assert q.items[3] == {"type": "result", "ok": True,
                          "info": {"duration": 1.0, "language": "de"}}


def test_exception_becomes_a_failed_result_with_type_and_trace(monkeypatch):
    def boom(**kw):
        raise ValueError("no such model")
    monkeypatch.setattr(voxtral_engine, "transcribe", boom)
    q = _Queue()
    voxtral_mp_worker.voxtral_proc_entrypoint({"audio_path": "x.wav"}, q)
    assert len(q.items) == 1
    res = q.items[0]
    assert res["type"] == "result" and res["ok"] is False
    assert res["error"] == "ValueError: no such model"
    assert "ValueError: no such model" in res["trace"]


def test_broken_log_and_progress_puts_do_not_fail_the_job(monkeypatch):
    monkeypatch.setattr(voxtral_engine, "transcribe", _fake_transcribe())
    q = _Queue(broken={"log", "progress"})
    voxtral_mp_worker.voxtral_proc_entrypoint({"audio_path": "x.wav"}, q)
    assert [m["type"] for m in q.items] == ["segment", "result"]
    assert q.items[-1]["ok"] is True


def test_broken_segment_put_fails_the_job_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(voxtral_engine, "transcribe", _fake_transcribe())
    q = _Queue(broken={"segment"})
    voxtral_mp_worker.voxtral_proc_entrypoint({"audio_path": "x.wav"}, q)
    assert q.items[-1]["type"] == "result" and q.items[-1]["ok"] is False
    assert q.items[-1]["error"].startswith("OSError")


def test_args_are_passed_through_by_name(monkeypatch):
    seen = {}
    def transcribe(**kw):
        seen.update(kw)
        return [], {}
    monkeypatch.setattr(voxtral_engine, "transcribe", transcribe)
    args = {"audio_path": "a.wav", "language_code": "de", "need_timestamps": False,
            "voxtral_repo": "r", "chunk_sec": 120, "corrections_path": "c.yml",
            "speaker_names": ["Mona"], "ram_reserve_gb": 4,
            "speaker_turns": [[0.0, 1.0, "S1"]]}
    voxtral_mp_worker.voxtral_proc_entrypoint(args, _Queue())
    assert seen["audio_path"] == "a.wav"
    assert seen["language"] == "de"
    assert seen["need_timestamps"] is False
    assert seen["voxtral_repo"] == "r"
    assert seen["chunk_sec"] == 120
    assert seen["corrections_path"] == "c.yml"
    assert seen["speaker_names"] == ["Mona"]
    assert seen["ram_reserve_gb"] == 4
    assert seen["speaker_turns"] == [[0.0, 1.0, "S1"]]


@pytest.mark.parametrize("inherited", [None, "0", "false"])
def test_hub_telemetry_is_off_before_the_engine_is_imported(inherited):
    """huggingface_hub reads HF_HUB_DISABLE_TELEMETRY once, at import, and the
    engine pulls it in (model downloads, the aligner). A clean interpreter in
    which importing voxtral_engine reports the value it was imported under,
    through the worker's own error path."""
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path
    probe = textwrap.dedent("""
        import importlib.abc, importlib.machinery, os, queue, sys

        class Report(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, name, path, target=None):
                if name == "noScribe.voxtral_engine":
                    return importlib.machinery.ModuleSpec(name, self)
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise RuntimeError("telemetry=%r" % os.environ.get("HF_HUB_DISABLE_TELEMETRY"))

        sys.meta_path.insert(0, Report())
        from noScribe.voxtral_mp_worker import voxtral_proc_entrypoint
        q = queue.Queue()
        voxtral_proc_entrypoint({}, q)
        print(q.get_nowait()["error"])
    """)
    env = dict(os.environ)
    env.pop("HF_HUB_DISABLE_TELEMETRY", None)
    if inherited is not None:
        env["HF_HUB_DISABLE_TELEMETRY"] = inherited
    proc = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0, proc.stderr
    assert "telemetry='1'" in proc.stdout, proc.stdout
