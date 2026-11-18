"""
Unit tests for nanometrix.calibration.parse_label().

parse_label() is the one place OCR noise from a burnt-in scale-bar label
turns into a physical length. Tesseract routinely garbles the unit glyph
(the micro sign most of all) and occasionally a digit, and parse_label()'s
job is to tolerate exactly that set of common OCR string shapes while still
rejecting text that carries no usable length. These tests pin down that
contract against synthetic OCR strings -- no image, OCR engine, or Tesseract
binary is needed, since parse_label() is a pure string -> (float, str)
function.
"""

from __future__ import annotations

import pytest

from nanometrix.calibration import parse_label


# --------------------------------------------------------------------------
# the exact formats called out in the backlog item
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_nm",
    [
        ("100 nm", 100.0),
        ("100nm", 100.0),
        ("1 um", 1_000.0),
        ("500 Å", 50.0),
    ],
)
def test_parse_label_backlog_formats(text, expected_nm):
    result = parse_label(text)
    assert result is not None
    value_nm, _pretty = result
    assert value_nm == pytest.approx(expected_nm)


# --------------------------------------------------------------------------
# unit conversion correctness, one per supported unit spelling
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected_nm",
    [
        ("100 nm", 100.0),          # already nanometres
        ("1 um", 1_000.0),          # ascii "u" for micro
        ("1 µm", 1_000.0),          # micro sign U+00B5
        ("1 μm", 1_000.0),          # greek mu U+03BC
        ("2.5 mm", 2_500_000.0),    # millimetres
        ("10 pm", 0.01),            # picometres
        ("500 Å", 50.0),            # angstrom sign
        ("500 A", 50.0),            # OCR's common angstrom-as-"A" misread
        ("500 a", 50.0),            # unit matching is case-insensitive too
    ],
)
def test_parse_label_unit_conversion_factors(text, expected_nm):
    value_nm, _pretty = parse_label(text)
    assert value_nm == pytest.approx(expected_nm)


def test_parse_label_unit_matching_is_case_insensitive():
    assert parse_label("100 NM") is not None
    assert parse_label("100 Nm") is not None
    assert parse_label("1 UM") is not None


# --------------------------------------------------------------------------
# decimal separators and whitespace noise
# --------------------------------------------------------------------------

def test_parse_label_accepts_comma_as_decimal_separator():
    """Some locales/instruments burn in a comma decimal point; parse_label
    must still recover the numeric value even though the pretty label keeps
    the original comma."""
    value_nm, pretty = parse_label("2,5 mm")
    assert value_nm == pytest.approx(2_500_000.0)
    assert pretty == "2,5 mm"


def test_parse_label_tolerates_extra_whitespace_and_surrounding_text():
    assert parse_label("  100   nm  ") == (100.0, "100 nm")
    assert parse_label("100nm scale bar") == (100.0, "100 nm")
    assert parse_label("scale: 100 nm") == (100.0, "100 nm")


# --------------------------------------------------------------------------
# OCR character-confusion correction (the whole reason this function exists)
# --------------------------------------------------------------------------

def test_parse_label_corrects_ocr_letter_o_as_zero():
    """Tesseract frequently reads a '0' glyph as a capital or lowercase 'O';
    parse_label pre-corrects both before matching digits."""
    assert parse_label("1OO nm") == (100.0, "100 nm")
    assert parse_label("1oo nm") == (100.0, "100 nm")


def test_parse_label_corrects_ocr_pipe_as_one():
    """A thin '1' is sometimes read as a vertical bar."""
    value_nm, _pretty = parse_label("|00 nm")
    assert value_nm == pytest.approx(100.0)


# --------------------------------------------------------------------------
# rejections: no usable length in the text
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "no units here 123",
        "100 xyz",
        "just some noise",
    ],
)
def test_parse_label_returns_none_when_no_match(text):
    assert parse_label(text) is None


def test_parse_label_rejects_zero_value():
    """A zero-length scale bar is physically meaningless, so parse_label
    must not treat '0 nm' as a successful read."""
    assert parse_label("0 nm") is None


def test_parse_label_rejects_unrecognized_unit():
    assert parse_label("100 ft") is None
    assert parse_label("100 inch") is None
