"""The pause threshold is stored as an option index and rendered in two places
(transcript header and job tooltip). Both go through pause_label, so the
index -> label mapping and its fallback are pinned here.
"""
import noScribe.main as m


def test_every_option_maps_to_its_own_label():
    # The labels are the identifiers the user picks in the dropdown and passes
    # to --pause, so they must survive rendering unchanged and in order.
    assert [m.pause_label(i) for i in range(len(m.PAUSE_OPTIONS))] == m.PAUSE_OPTIONS


def test_a_rendered_label_maps_back_to_its_index():
    """The header shows what the dropdown offers, so the two have to agree --
    which is why they share one list instead of keeping a copy each."""
    for i, label in enumerate(m.PAUSE_OPTIONS):
        assert m.PAUSE_OPTIONS.index(m.pause_label(i)) == i


def test_out_of_range_and_non_int_fall_back_to_str():
    # Guards against an IndexError in the transcript header if the stored
    # config value ever drifts out of the option list.
    assert m.pause_label(99) == '99'
    assert m.pause_label('2sec+') == '2sec+'
    assert m.pause_label(None) == 'None'
    # bool is a subclass of int; without the explicit check True would index
    # the list and render as '1sec+'.
    assert m.pause_label(True) == 'True'
    assert m.pause_label(False) == 'False'
