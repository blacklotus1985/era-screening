"""
Tests for the file readers of experiments/run_screening.py.

Regression guard for a real bug: the CLI used one stripped-lines reader for
both contexts and probe vocabulary.  In BPE tokenizers the leading space is
part of the token (on GPT-Neo, "man" is id 805 while " man" is id 582), so
stripping probe words silently measured different tokens than the ones the
auditor wrote in the file.

The script imports torch-dependent modules only inside main(), so it can be
loaded here via importlib in a torch-free environment (same pattern as
test_compare_curves.py).
"""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "experiments" / "run_screening.py"
_spec = importlib.util.spec_from_file_location("run_screening", _SCRIPT)
run_screening = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_screening)


def test_context_reader_strips_whitespace(tmp_path):
    p = tmp_path / "contexts.txt"
    p.write_text("  A CEO is a  \n\n B \n", encoding="utf-8")
    assert run_screening.read_context_lines(str(p)) == ["A CEO is a", "B"]


def test_probe_reader_preserves_leading_space(tmp_path):
    """The whole point of the split: ' man' must stay ' man'."""
    p = tmp_path / "probes.txt"
    p.write_text(" man\n woman\nnurse\n", encoding="utf-8")
    assert run_screening.read_probe_vocab_lines(str(p)) == [" man", " woman", "nurse"]


def test_probe_reader_strips_line_endings_and_blank_lines(tmp_path):
    p = tmp_path / "probes.txt"
    # CRLF endings and a whitespace-only line (must be skipped, not kept).
    p.write_text(" man\r\n\n   \n leader\r\n", encoding="utf-8")
    assert run_screening.read_probe_vocab_lines(str(p)) == [" man", " leader"]
