"""The pause threshold is stored as an option index and rendered in two places
(transcript header and job tooltip). Both go through pause_label, so the
index -> label mapping and its fallback are pinned here.
"""
import pytest

pytest.importorskip("tkinter")  # noScribe.main pulls in the GUI stack

import noScribe.main as m


def test_known_indices_map_to_their_label():
    assert m.pause_label(1) == '1sec+'
    assert m.pause_label(3) == '3sec+'


def test_out_of_range_and_non_int_fall_back_to_str():
    # Guards against an IndexError in the transcript header if the stored
    # config value ever drifts out of the option list.
    assert m.pause_label(99) == '99'
    assert m.pause_label('2sec+') == '2sec+'
    assert m.pause_label(None) == 'None'
