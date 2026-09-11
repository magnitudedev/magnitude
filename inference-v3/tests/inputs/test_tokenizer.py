import os
import unicodedata
from pathlib import Path

import pytest
from tokenizers import Tokenizer as ReferenceTokenizer

from magnitude_engine.blueprints import inputs
from magnitude_engine.blueprints import weights as containers
from magnitude_engine.composition import build, dumps, loads
from magnitude_engine.data import TokenId
from magnitude_engine.inputs.tokenizer import BPEConfig, ByteBPETokenizer, PieceKind, SpecialTokens
from magnitude_engine.weights.identity import ArtifactIdentity


def byte_tokenizer():
    # Independently construct the GPT byte alphabet for a merge-free fixture.
    included = {*range(33, 127), *range(161, 173), *range(174, 256)}
    missing = [value for value in range(256) if value not in included]
    pieces = tuple(
        chr(value) if value in included else chr(256 + missing.index(value)) for value in range(256)
    )
    config = BPEConfig(
        artifact_identity=ArtifactIdentity("0" * 64),
        pieces=(*pieces, "<control>", "<ordinary>"),
        kinds=(*((PieceKind.NORMAL,) * 256), PieceKind.CONTROL, PieceKind.USER_DEFINED),
        merges=(),
        pattern=r".+|\s",
        normalize_nfc=True,
        stop_tokens=frozenset({TokenId(256)}),
    )
    assert BPEConfig.model_validate_json(config.model_dump_json()) == config
    return ByteBPETokenizer(config)


def test_incremental_utf8_special_handling_and_finalization():
    tokenizer = byte_tokenizer()
    text = "a e\u0301 नमस्ते 👩🏽‍💻 日本語\n"
    tokens = tokenizer.encode(text)
    decoder = tokenizer.decoder()
    parts = [decoder.push(token) for token in tokens]
    assert "�" not in "".join(parts)
    assert "".join(parts) + decoder.finish() == unicodedata.normalize("NFC", text)
    assert "" in parts  # multi-byte characters span multiple fixture tokens
    with pytest.raises(RuntimeError, match="finished"):
        decoder.push(TokenId(1))
    with pytest.raises(RuntimeError, match="finished"):
        decoder.finish()
    incomplete = tokenizer.decoder()
    assert incomplete.push(TokenId(0xE2)) == ""
    assert incomplete.finish() == "�"
    assert tokenizer.encode("<control><ordinary>") == (256, 257)
    literal = tokenizer.encode("<control><ordinary>", special=SpecialTokens.LITERAL)
    assert 256 not in literal and literal[-1] == 257
    assert tokenizer.decode((TokenId(256), TokenId(257))) == "<ordinary>"
    assert (
        tokenizer.decode((TokenId(256), TokenId(257)), skip_control=False) == "<control><ordinary>"
    )
    with pytest.raises(ValueError, match="vocabulary"):
        tokenizer.piece(TokenId(-1))


@pytest.mark.model
def test_pinned_gguf_tokenization_matches_published_qwen_reference():
    source = os.environ.get("MAGNITUDE_TEST_GGUF")
    reference_path = os.environ.get("MAGNITUDE_TEST_TOKENIZER_REFERENCE")
    if source is None or reference_path is None:
        pytest.skip("set MAGNITUDE_TEST_GGUF and MAGNITUDE_TEST_TOKENIZER_REFERENCE")
    recipe = inputs.ByteBPE(config=inputs.Qwen35Tokenization(artifact=containers.GGUF(path=source)))
    with build(loads(dumps(recipe))) as tokenizer:
        pass
    reference = ReferenceTokenizer.from_file(str(Path(reference_path)))
    examples = (
        "The capital of France is",
        "hello world 12345",
        "e\u0301lan हिन्दी 日本語",
        "नमस्ते ప్రపంచం ภาษาไทย العربية",
        "\t\r\n  abc   \n",
        "👩🏽‍💻 🇫🇷 🦀",
        "can't I'LL we're FooBAR",
        "<think>\n<|im_start|>assistant",
        "x\x00y\x01z",
        "def f(x: int):\n    return x**2  # αβγ",
        "a" * 257 + " 1234567890 " * 19,
        '<tool_call>{"name":"f"}</tool_call><|im_end|>',
        "<|fim_prefix|>x<|fim_suffix|>",
    )
    for example in examples:
        actual = tokenizer.encode(example)
        expected = reference.encode(example, add_special_tokens=False).ids
        assert actual == tuple(expected), repr(example)
        assert tokenizer.decode(actual, skip_control=False) == reference.decode(
            expected, skip_special_tokens=False
        )
    assert tokenizer.stop_tokens == {248044, 248046}
