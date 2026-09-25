from noScribe.server.storage import cleanup_stale_audio, prepare_runtime_directory


def test_runtime_cleanup_only_removes_server_generated_audio(tmp_path):
    stale_opus = tmp_path / "upload-abc.opus"
    stale_wav = tmp_path / "upload-abc.wav"
    unrelated = tmp_path / "keep.txt"
    stale_opus.write_bytes(b"sensitive")
    stale_wav.write_bytes(b"sensitive")
    unrelated.write_text("keep", encoding="utf-8")

    cleanup_stale_audio(tmp_path)

    assert not stale_opus.exists()
    assert not stale_wav.exists()
    assert unrelated.exists()


def test_runtime_directory_can_be_prepared_without_tmpfs_requirement(tmp_path):
    runtime = tmp_path / "runtime"

    prepare_runtime_directory(runtime, require_tmpfs=False)

    assert runtime.is_dir()
