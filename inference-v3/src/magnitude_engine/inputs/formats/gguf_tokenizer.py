"""Interpret container tokenizer metadata into a container-independent BPE contract.

Tokenizer metadata travels in the same files as the weights, but it is an input
concern: nothing here reaches residency or a kernel.
"""

from pathlib import Path

from pydantic import TypeAdapter

from magnitude_engine.data import Record, TokenId
from magnitude_engine.inputs.tokenizer import BPEConfig, PieceKind
from magnitude_engine.weights.formats.gguf import GGUFFormat
from magnitude_engine.weights.formats.mlx_safetensors import MLXFormat


def qwen35_tokenizer(artifact: GGUFFormat) -> BPEConfig:
    directory = artifact.directory
    if (
        directory.value("tokenizer.ggml.model") != "gpt2"
        or directory.value("tokenizer.ggml.pre") != "qwen35"
    ):
        raise ValueError("tokenizer binding requires Qwen3.5 byte BPE metadata")
    metadata = {item.name: item.value for item in directory.metadata}
    if metadata.get("tokenizer.ggml.add_bos_token", False) or metadata.get(
        "tokenizer.ggml.add_eos_token", False
    ):
        raise ValueError("Qwen3.5 input preparation does not insert implicit token markers")
    strings = TypeAdapter(tuple[str, ...])
    pieces = strings.validate_python(directory.value("tokenizer.ggml.tokens"), strict=True)
    kinds = TypeAdapter(tuple[int, ...]).validate_python(
        directory.value("tokenizer.ggml.token_type"), strict=True
    )
    merges = strings.validate_python(directory.value("tokenizer.ggml.merges"), strict=True)
    eos = directory.value("tokenizer.ggml.eos_token_id")
    pairs = []
    for entry in merges:
        parts = entry.split(" ")
        if len(parts) != 2:
            raise ValueError("tokenizer merge entries must contain two byte-BPE pieces")
        pairs.append((parts[0], parts[1]))
    if type(eos) is not int:
        raise ValueError("tokenizer EOS must be a token ID")
    stops = {TokenId(eos)}
    stops.update(
        TokenId(i) for i, piece in enumerate(pieces) if piece in ("<|endoftext|>", "<|im_end|>")
    )
    # Qualified against the original Qwen tokenizer.json and llama.cpp's
    # QWEN35 pre-tokenizer. Combining marks are part of a word in this family.
    pattern = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
        r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
    return BPEConfig(
        artifact_identity=artifact.identity,
        pieces=pieces,
        kinds=tuple(PieceKind(k) for k in kinds),
        merges=tuple(pairs),
        pattern=pattern,
        normalize_nfc=True,
        stop_tokens=frozenset(stops),
    )


class TokenizerArtifact(Record):
    """Immutable tokenizer and chat metadata, without mapped-file ownership."""

    config: BPEConfig
    chat_template: str

    @classmethod
    def interpret(cls, artifact: GGUFFormat):
        config = qwen35_tokenizer(artifact)
        template = artifact.directory.value("tokenizer.chat_template")
        if not isinstance(template, str) or not template:
            raise ValueError("GGUF must contain its chat template")
        return cls(config=config, chat_template=template)

    @classmethod
    def load(cls, path: Path):
        """Read tokenizer metadata from whichever container the path names."""
        artifact = MLXFormat(str(path)) if path.is_dir() else GGUFFormat(str(path))
        try:
            if isinstance(artifact, MLXFormat):
                return mlx_tokenizer(artifact)
            return cls.interpret(artifact)
        finally:
            artifact.close()


def mlx_tokenizer(artifact: MLXFormat) -> TokenizerArtifact:
    """Interpret the converted Qwen tokenizer without introducing another tokenizer runtime."""
    import json

    from magnitude_engine.platform.storage import FileSource

    with FileSource(artifact.path / "tokenizer.json") as source:
        data = json.loads(source.read(0, source.size))
    model = data["model"]
    if model["type"] != "BPE" or model.get("byte_fallback") or model.get("unk_token"):
        raise ValueError("MLX Qwen tokenizer requires byte BPE without unknown replacement")
    split, byte_level = data["pre_tokenizer"]["pretokenizers"]
    if (
        data["normalizer"] != {"type": "NFC"}
        or split["type"] != "Split"
        or split["behavior"] != "Isolated"
        or split["invert"]
        or byte_level
        != {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": False,
            "use_regex": False,
        }
    ):
        raise ValueError("unsupported converted Qwen pre-tokenization")
    vocabulary = TypeAdapter(dict[str, int]).validate_python(model["vocab"], strict=True)
    by_id = {identity: (piece, PieceKind.NORMAL) for piece, identity in vocabulary.items()}
    if len(by_id) != len(vocabulary):
        raise ValueError("duplicate tokenizer vocabulary IDs")
    for token in data["added_tokens"]:
        if token["single_word"] or token["lstrip"] or token["rstrip"] or token["normalized"]:
            raise ValueError("unsupported added-token matching policy")
        by_id[token["id"]] = (
            token["content"],
            PieceKind.CONTROL if token["special"] else PieceKind.USER_DEFINED,
        )
    if set(by_id) != set(range(max(by_id) + 1)):
        raise ValueError("converted tokenizer IDs must be contiguous")
    pieces = tuple(by_id[i][0] for i in range(len(by_id)))
    config = BPEConfig(
        artifact_identity=artifact.identity,
        pieces=pieces,
        kinds=tuple(by_id[i][1] for i in range(len(by_id))),
        merges=TypeAdapter(tuple[tuple[str, str], ...]).validate_python(
            model["merges"], strict=False
        ),
        pattern=split["pattern"]["Regex"],
        normalize_nfc=True,
        stop_tokens=frozenset(
            TokenId(i) for i, piece in enumerate(pieces) if piece in ("<|im_end|>", "<|endoftext|>")
        ),
    )
    with FileSource(artifact.path / "chat_template.jinja") as source:
        template = source.read(0, source.size).decode("utf-8")
    if not template:
        raise ValueError("MLX artifact must contain its chat template")
    return TokenizerArtifact(config=config, chat_template=template)
