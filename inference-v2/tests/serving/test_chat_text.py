import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from magnitude_engine.serving.formats import format_for
from magnitude_engine.serving.parsing import OutputParser, TextDelta
from magnitude_engine.serving.text import StopText, TokenText


def test_incremental_byte_tokenizer_holds_incomplete_unicode_and_preserves_markers():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<think>", "</think>"])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    text = "<think>café 😀 世界</think> one , two\n"
    tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
    for width in (1, 2, 5, len(tokens)):
        decoder = TokenText(tokenizer)
        parts = [decoder.feed(tokens[i : i + width]) for i in range(0, len(tokens), width)]
        assert not any("\ufffd" in part for part in parts)
        parts.append(decoder.feed((), final=True))
        assert "".join(parts) == text
        with pytest.raises(RuntimeError, match="closed"):
            decoder.feed(())


@pytest.mark.parametrize(
    "text,stops,expected,matched",
    [
        ("hello<STOP>hidden", ("<STOP>",), "hello", "<STOP>"),
        ("hello<ST", ("<STOP>",), "hello<ST", None),
        ("abc", ("abc", "b"), "a", "b"),
        ("abc", ("ab", "abc"), "", "ab"),
        ("世界終later", ("終",), "世界", "終"),
    ],
)
def test_stop_matching_is_chunk_invariant_with_bounded_pending_text(text, stops, expected, matched):
    for split in range(len(text) + 1):
        filter = StopText(stops)
        output = []
        for part in (text[:split], text[split:]):
            output.append(filter.feed(part))
            assert len(filter.pending) < max(map(len, stops))
        output.append(filter.feed("", final=True))
        assert "".join(output) == expected and filter.matched == matched


def test_length_truncation_preserves_raw_call_without_inventing_a_successful_tool():
    parser = OutputParser(format_for("qwen3"), [])
    text = '<tool_call>{"name":"f","arguments":'
    assert parser.feed(text) == []
    assert parser.feed("", final=True, truncated=True) == [TextDelta("content", text)]
