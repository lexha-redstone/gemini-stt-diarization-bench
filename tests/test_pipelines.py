"""Unit tests for SingleStepPipeline and turn parsing."""

import pytest
from src.pipelines.single_step import SingleStepPipeline
from src.models import Turn


def test_parse_turns_standard_format():
    """Tests parsing of standard 'Speaker 0: ...' format."""
    raw_text = (
        "Speaker 0: हाँ, मैं बोल रहा हूँ।\n"
        "Speaker 1: नमस्कार, क्या मेरी बात शर्मा जी से हो रही है?\n"
        "Speaker 0: जी, कहिए।"
    )
    turns = SingleStepPipeline.parse_turns(raw_text)
    assert len(turns) == 3
    assert turns[0].speaker == "Speaker 0"
    assert turns[0].text == "हाँ, मैं बोल रहा हूँ।"
    assert turns[1].speaker == "Speaker 1"
    assert turns[1].text == "नमस्कार, क्या मेरी बात शर्मा जी से हो रही है?"
    assert turns[2].speaker == "Speaker 0"
    assert turns[2].text == "जी, कहिए।"


def test_parse_turns_markdown_and_brackets():
    """Tests parsing when output includes markdown bold or bracketed speakers."""
    raw_text = (
        "**Speaker 1:** नमस्कार सर\n"
        "[Speaker 2]: जी नमस्ते, कौन बोल रहा है?\n"
        "**Speaker 1:** मैं बैंक से बात कर रहा हूँ।"
    )
    turns = SingleStepPipeline.parse_turns(raw_text)
    assert len(turns) == 3
    assert turns[0].speaker == "Speaker 1"
    assert turns[0].text == "नमस्कार सर"
    assert turns[1].speaker == "Speaker 2"
    assert turns[1].text == "जी नमस्ते, कौन बोल रहा है?"
    assert turns[2].speaker == "Speaker 1"
    assert turns[2].text == "मैं बैंक से बात कर रहा हूँ।"


def test_parse_turns_multiline_utterance():
    """Tests that subsequent unlabelled lines are attached to the current speaker."""
    raw_text = (
        "Speaker 0: पहला वाक्य।\n"
        "दूसरा वाक्य भी इसी वक्ता का है।\n"
        "Speaker 1: यह दूसरे वक्ता का उत्तर है।"
    )
    turns = SingleStepPipeline.parse_turns(raw_text)
    assert len(turns) == 2
    assert turns[0].speaker == "Speaker 0"
    assert "पहला वाक्य।" in turns[0].text
    assert "दूसरा वाक्य भी इसी वक्ता का है।" in turns[0].text
    assert turns[1].speaker == "Speaker 1"


def test_parse_turns_hindi_speaker_prefix():
    """Tests parsing with Devanagari speaker prefixes."""
    raw_text = (
        "वक्ता 1: नमस्ते सर।\n"
        "वक्ता 2: हाँ जी बताइए।"
    )
    turns = SingleStepPipeline.parse_turns(raw_text)
    assert len(turns) == 2
    assert turns[0].speaker == "वक्ता 1"
    assert turns[0].text == "नमस्ते सर।"
    assert turns[1].speaker == "वक्ता 2"
    assert turns[1].text == "हाँ जी बताइए।"


def test_parse_turns_empty_or_whitespace():
    """Tests that empty or whitespace text safely yields an empty list."""
    assert SingleStepPipeline.parse_turns("") == []
    assert SingleStepPipeline.parse_turns("   \n\n  ") == []
