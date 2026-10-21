"""Guard for #368: the translation folder handed to python-i18n must stay valid.

python-i18n reads every folder in its load path again for keys it has not
memoized yet (a key missing from the current locale, or the fallback to "en").
main.py and the Whisper worker used to add the folder from
``impres.as_file(impres.files("trans"))``, a temporary copy that is deleted
when the ``with`` block ends, so the first such lookup crashed with
FileNotFoundError.
"""
import os
from pathlib import Path

import i18n
import pytest
from i18n import t

from noScribe import utils

REPO = Path(__file__).resolve().parent.parent
MISSING_KEY = "key_that_is_in_no_translation_file"


def _without_comments(text):
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


@pytest.fixture
def clean_i18n():
    """Configure python-i18n like main.py and restore its global state after."""
    settings = dict(i18n.config.settings)
    load_path = list(i18n.load_path)
    container = dict(i18n.translations.container)
    i18n.load_path.clear()
    i18n.translations.container.clear()
    i18n.set("filename_format", "{locale}.{format}")
    i18n.set("enable_memoization", True)
    i18n.set("fallback", "en")
    yield
    i18n.config.settings.clear()
    i18n.config.settings.update(settings)
    i18n.load_path[:] = load_path
    i18n.translations.container.clear()
    i18n.translations.container.update(container)


def test_translations_dir_is_a_real_folder():
    path = utils.translations_dir()
    assert os.path.isfile(os.path.join(path, "noScribe.en.yml"))


@pytest.mark.parametrize("locale", ["en", "de"])
def test_missing_key_falls_back_to_the_key(clean_i18n, locale):
    i18n.set("locale", locale)
    i18n.load_path.append(utils.translations_dir())
    assert t("app_header") != "app_header"

    assert t(MISSING_KEY) == MISSING_KEY


@pytest.mark.parametrize("module", ["main.py", "whisper_mp_worker.py"])
def test_trans_folder_is_not_a_temporary_copy(module):
    source = _without_comments(
        (REPO / "noScribe" / module).read_text(encoding="utf-8"))
    assert 'as_file(impres.files("trans"))' not in source
    assert "translations_dir()" in source
