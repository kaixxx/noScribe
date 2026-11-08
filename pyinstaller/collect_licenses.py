"""Collect installed packages' license documents for a PyInstaller build.

Selection uses the source paths in PyInstaller's analysis, rather than all
packages in the build environment. No license texts are downloaded or inferred.
"""

from importlib import metadata
import logging
import os
from pathlib import Path, PurePosixPath
import re


logger = logging.getLogger(__name__)
LICENSE_NAME = re.compile(
    r"^(?:licen[sc]es?|notice|copying|copyright|eula)(?:$|[._-])"
    r"|^third[-_ ]?party[-_ ]?(?:licen[sc]es?|notices?)(?:$|[._-])",
    re.IGNORECASE,
)


def _source_key(path):
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _document_path(record, declared_files):
    """Return a safe relative path for a license document, or None."""
    path = PurePosixPath(str(record).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in str(path):
        return None
    declared = any(
        str(path) == name or str(path).endswith("/" + name)
        for name in declared_files
    )
    if (
        declared
        or LICENSE_NAME.match(path.name)
        or any(
            part.lower() in {"license", "licenses", "licence", "licences"}
            for part in path.parts[:-1]
        )
    ):
        return path
    return None


def collect_license_files(entries, output_dir):
    """Return DATA entries, including a generated package/version overview.

    ``entries`` is a sequence of PyInstaller (destination, source, type) tuples.
    The returned tuples can be passed directly to COLLECT. Original document paths
    are preserved inside a separate directory for each distribution and version.
    """
    included_sources = {
        _source_key(source) for _, source, _ in entries if source
    }
    packages = []
    for distribution in metadata.distributions():
        records = list(distribution.files or ())
        if not any(
            _source_key(distribution.locate_file(record)) in included_sources
            for record in records
        ):
            continue
        info = distribution.metadata
        name = info.get("Name") or "unknown"
        version = distribution.version
        directory = re.sub(r"[^a-zA-Z0-9_.+-]", "_", f"{name}-{version}").strip(".")
        declared_files = [
            name.replace("\\", "/")
            for name in info.get_all("License-File", ())
        ]
        documents = []
        for record in records:
            relative = _document_path(record, declared_files)
            if relative is None:
                continue
            source = Path(distribution.locate_file(record))
            if source.is_file():
                destination = PurePosixPath(directory) / relative
                documents.append((str(destination), str(source)))
        packages.append((name, version, sorted(set(documents))))

    data = []
    overview = [
        "Third-party license files",
        "",
        "Generated from installed distributions represented in this build.",
        "Only documents supplied by those distributions are collected.",
        "This does not establish completeness for native libraries, models,",
        "or embedded applications such as noScribeEdit.",
        "",
    ]
    missing = []
    for name, version, documents in sorted(packages):
        overview.append(f"{name} {version}")
        for destination, source in documents:
            overview.append(f"  {destination}")
            data.append((f"third_party_licenses/{destination}", source, "DATA"))
        if not documents:
            overview.append("  No separate license document found in the installed distribution.")
            missing.append(name)
        overview.append("")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overview_path = output_dir / "third_party_licenses.txt"
    overview_path.write_text("\n".join(overview), encoding="utf-8")
    logger.info("Collected %d license documents from %d distributions", len(data), len(packages))
    if missing:
        logger.warning(
            "No separate license documents found for: %s. See third_party_licenses/README.txt.",
            ", ".join(missing),
        )
    data.append(("third_party_licenses/README.txt", str(overview_path), "DATA"))
    return data
