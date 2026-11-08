"""Builds retain license documents without collecting unrelated packages."""

import csv
from importlib import metadata
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "collect_licenses", Path(__file__).resolve().parents[1] / "pyinstaller/collect_licenses.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


@pytest.fixture(autouse=True)
def project_license_documents(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "CUDA-EXCEPTION.txt").write_text("Scoped CUDA permission", encoding="utf-8")
    nvidia = root / "third_party_licenses/nvidia"
    nvidia.mkdir(parents=True)
    (nvidia / "LICENSE.txt").write_text("Original NVIDIA license", encoding="utf-8")
    (nvidia / "README.md").write_text("Snapshot provenance", encoding="utf-8")
    monkeypatch.setattr(collector, "PROJECT_ROOT", root)


def make_distribution(site, name, files, *, license_file=None):
    info = site / f"{name}-1.2.dist-info"
    info.mkdir(parents=True)
    text = f"Metadata-Version: 2.4\nName: {name}\nVersion: 1.2\n"
    if license_file:
        text += f"License-File: {license_file}\n"
    (info / "METADATA").write_text(text, encoding="utf-8")
    for relative, content in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    with (info / "RECORD").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        for relative in [*files, f"{info.name}/METADATA"]:
            writer.writerow([relative, "", ""])
    return metadata.PathDistribution(info)


def test_collects_metadata_and_package_documents_for_included_distribution(tmp_path, monkeypatch):
    site = tmp_path / "site"
    used = make_distribution(site, "used", {
        "shared/used.py": "# included module",
        "used-1.2.dist-info/licenses/LICENSE": "metadata license",
        "used-1.2.dist-info/licenses/vendor/attribution.txt": "vendor attribution",
        "shared/ThirdPartyNotices.txt": "third-party notices",
        "shared/README.md": "ordinary readme",
    })
    unused = make_distribution(site, "unused", {
        "shared/unused.py": "# same namespace, not included",
        "unused-1.2.dist-info/LICENSE": "unrelated license",
    })
    monkeypatch.setattr(collector.metadata, "distributions", lambda: [unused, used])
    entries = [("shared.used", str(site / "shared/used.py"), "PYMODULE")]

    data = collector.collect_license_files(entries, tmp_path / "work")
    destinations = {destination for destination, _, _ in data}
    assert destinations == {
        "CUDA-EXCEPTION.txt",
        "third_party_licenses/used-1.2/used-1.2.dist-info/licenses/LICENSE",
        "third_party_licenses/used-1.2/used-1.2.dist-info/licenses/vendor/attribution.txt",
        "third_party_licenses/used-1.2/shared/ThirdPartyNotices.txt",
        "third_party_licenses/README.txt",
    }
    overview = Path(data[-1][1]).read_text(encoding="utf-8")
    assert "used 1.2" in overview
    assert "unused" not in overview
    assert all(Path(source).is_file() and kind == "DATA" for _, source, kind in data)


def test_honors_declared_license_filename_for_binary_distribution(tmp_path, monkeypatch):
    site = tmp_path / "site"
    distribution = make_distribution(site, "native", {
        "native/extension.pyd": "binary stand-in",
        "native-1.2.dist-info/AUTHORS": "required attribution",
    }, license_file="AUTHORS")
    monkeypatch.setattr(collector.metadata, "distributions", lambda: [distribution])

    data = collector.collect_license_files(
        [("native/extension.pyd", str(site / "native/extension.pyd"), "BINARY")],
        tmp_path / "work",
    )
    assert any(destination.endswith("native-1.2.dist-info/AUTHORS") for destination, _, _ in data)


def test_records_missing_documents_and_does_not_escape_output(tmp_path, monkeypatch, caplog):
    site = tmp_path / "site"
    distribution = make_distribution(site, "incomplete", {
        "incomplete/__init__.py": "# included",
        "../LICENSE": "outside distribution",
    })
    monkeypatch.setattr(collector.metadata, "distributions", lambda: [distribution])

    data = collector.collect_license_files(
        [("incomplete", str(site / "incomplete/__init__.py"), "PYMODULE")],
        tmp_path / "work",
    )
    assert len(data) == 2
    overview = Path(data[-1][1]).read_text(encoding="utf-8")
    assert "incomplete 1.2" in overview
    assert "No separate license document found" in overview
    assert "incomplete" in caplog.text


@pytest.mark.parametrize("library", ["cublas64_12.dll", "libcudnn.so.9.10.2", "torch_cuda.dll"])
def test_supplements_are_included_only_for_nvidia_libraries(tmp_path, monkeypatch, library):
    monkeypatch.setattr(collector.metadata, "distributions", lambda: [])
    data = collector.collect_license_files(
        [(library, str(tmp_path / library), "BINARY")], tmp_path / "work"
    )
    destinations = {destination for destination, _, _ in data}
    assert "CUDA-EXCEPTION.txt" in destinations
    assert ("third_party_licenses/nvidia/LICENSE.txt" in destinations) == (library != "torch_cuda.dll")
    assert ("third_party_licenses/nvidia/README.md" in destinations) == (library != "torch_cuda.dll")


def test_cuda_build_requires_supplemental_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.metadata, "distributions", lambda: [])
    monkeypatch.setattr(collector, "PROJECT_ROOT", tmp_path)
    (tmp_path / "CUDA-EXCEPTION.txt").write_text("Scoped permission", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="NVIDIA license documents missing"):
        collector.collect_license_files(
            [("libcudart.so.12", str(tmp_path / "libcudart.so.12"), "BINARY")],
            tmp_path / "work",
        )
