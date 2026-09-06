from io import StringIO

from magnitude_engine.chat.terminal import Terminal
from magnitude_engine.engine.delivery import Finished, PrefillProgress
from magnitude_engine.serving.parsing import TextDelta


def test_terminal_uses_completed_progress_and_excludes_first_token_from_decode_rate():
    output = StringIO()
    terminal = Terminal(output)
    terminal.begin()
    terminal.progress(PrefillProgress(99, 99, 49, 1_000_000_000))
    terminal.text(TextDelta('content', 'Hello'))
    assert output.getvalue().endswith('Assistant: Hello')
    terminal.finish(Finished(
        'stop', 100, 5, 49, 6, 4, 100_000_000, 1_200_000_000, 3_200_000_000,
        prefill_ns=1_000_000_000, decode_ns=2_100_000_000, first_decode_ns=100_000_000,
    ), 3_400_000_000)
    text = output.getvalue()
    assert 'Prefill 99/99 tokens (100%)' in text
    assert 'TTFT 1.200s' in text
    assert 'Decode after first token: 2.000s · 2.0 tok/s' in text
    assert 'Draft acceptance: 4/6 (66.7%)' in text
    assert '\x1b' not in text


def test_single_output_does_not_report_an_infinite_decode_rate():
    output = StringIO()
    Terminal(output).finish(Finished('length', 1, 1, 0, 0, 0, 0, 1, 1,
                                   decode_ns=1, first_decode_ns=1), 2)
    assert 'Decode after first token: 0.000s · n/a' in output.getvalue()
