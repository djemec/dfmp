from dfmp.config import DEFAULT_COLOR, LABEL_FONTSIZE, VALUE_FONTSIZE, TITLE_FONTSIZE, MIN_HEIGHT_FRAC


def test_default_color():
    assert DEFAULT_COLOR == '#5B9A8B'


def test_label_fontsize():
    assert LABEL_FONTSIZE == 11


def test_value_fontsize():
    assert VALUE_FONTSIZE == 10


def test_title_fontsize():
    assert TITLE_FONTSIZE == 13


def test_min_height_frac():
    assert MIN_HEIGHT_FRAC == 0.45
    assert 0 < MIN_HEIGHT_FRAC < 1
