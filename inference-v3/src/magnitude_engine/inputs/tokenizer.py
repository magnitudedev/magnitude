"""Byte BPE text contract and incremental UTF-8 publication decoding."""

from __future__ import annotations

import codecs
from abc import ABC, abstractmethod
from enum import IntEnum, StrEnum

from pydantic import model_validator

from magnitude_engine.data import Record, TokenId
from magnitude_engine.weights.identity import ArtifactIdentity


class PieceKind(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class SpecialTokens(StrEnum):
    RECOGNIZE = "recognize"
    LITERAL = "literal"


class BPEConfig(Record):
    artifact_identity: ArtifactIdentity
    pieces: tuple[str, ...]
    kinds: tuple[PieceKind, ...]
    merges: tuple[tuple[str, str], ...]
    pattern: str
    normalize_nfc: bool
    stop_tokens: frozenset[TokenId]

    @model_validator(mode="after")
    def validate_vocabulary(self):
        if not self.pieces or len(self.pieces) != len(self.kinds):
            raise ValueError("vocabulary pieces and kinds must align")
        if len(set(self.pieces)) != len(self.pieces):
            raise ValueError("vocabulary contains duplicate pieces")
        if any(not 0 <= token < len(self.pieces) for token in self.stop_tokens):
            raise ValueError("stop token is outside the vocabulary")
        return self


class Tokenizer(ABC):
    @property
    @abstractmethod
    def vocabulary(self) -> int: ...

    @property
    @abstractmethod
    def stop_tokens(self) -> frozenset[TokenId]: ...

    @abstractmethod
    def encode(
        self, text: str, *, special: SpecialTokens = SpecialTokens.RECOGNIZE
    ) -> tuple[TokenId, ...]: ...

    @abstractmethod
    def piece(self, token: TokenId, *, skip_control: bool = True) -> bytes: ...

    def decoder(self, *, skip_control: bool = True) -> TokenDecoder:
        return TokenDecoder(self, skip_control=skip_control)

    def decode(self, tokens: tuple[TokenId, ...], *, skip_control: bool = True) -> str:
        decoder = self.decoder(skip_control=skip_control)
        return "".join(decoder.push(token) for token in tokens) + decoder.finish()


class TokenDecoder:
    def __init__(self, tokenizer: Tokenizer, *, skip_control: bool):
        self.tokenizer, self.skip_control = tokenizer, skip_control
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._finished = False

    def push(self, token: TokenId) -> str:
        if self._finished:
            raise RuntimeError("text decoder is finished")
        return self._decoder.decode(
            self.tokenizer.piece(token, skip_control=self.skip_control), final=False
        )

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("text decoder is finished")
        self._finished = True
        return self._decoder.decode(b"", final=True)


class ByteBPETokenizer(Tokenizer):
    def __init__(self, config: BPEConfig):
        from tokenizers import AddedToken, Regex, decoders, normalizers, pre_tokenizers
        from tokenizers import Tokenizer as NativeTokenizer
        from tokenizers.models import BPE

        self.config = config
        # GPT byte-to-Unicode alphabet. All byte values remain recoverable even
        # when a single vocabulary token ends inside a UTF-8 character.
        byte_values = [*range(33, 127), *range(161, 173), *range(174, 256)]
        codepoints = list(byte_values)
        next_codepoint = 256
        for value in range(256):
            if value not in byte_values:
                byte_values.append(value)
                codepoints.append(next_codepoint)
                next_codepoint += 1
        alphabet = {
            chr(codepoint): value for codepoint, value in zip(codepoints, byte_values, strict=True)
        }
        pieces = []
        for piece, kind in zip(config.pieces, config.kinds, strict=True):
            if kind == PieceKind.NORMAL:
                try:
                    pieces.append(bytes(alphabet[character] for character in piece))
                except KeyError as error:
                    raise ValueError("normal BPE piece is outside the byte alphabet") from error
            elif kind in (PieceKind.CONTROL, PieceKind.USER_DEFINED, PieceKind.UNUSED):
                pieces.append(piece.encode("utf-8"))
            else:
                raise ValueError(f"byte BPE does not implement piece kind {kind.name}")
        self._pieces = tuple(pieces)
        model = BPE(
            vocab={piece: i for i, piece in enumerate(config.pieces)},
            merges=list(config.merges),
            unk_token=None,
            fuse_unk=False,
            byte_fallback=False,
        )
        self._encoders = {}
        for handling in SpecialTokens:
            encoder = NativeTokenizer(model)
            if config.normalize_nfc:
                encoder.normalizer = normalizers.NFC()
            encoder.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(Regex(config.pattern), behavior="isolated"),
                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
                ]
            )
            encoder.decoder = decoders.ByteLevel()
            for token, (piece, kind) in enumerate(zip(config.pieces, config.kinds, strict=True)):
                if kind == PieceKind.USER_DEFINED or (
                    kind == PieceKind.CONTROL and handling == SpecialTokens.RECOGNIZE
                ):
                    encoder.add_tokens(
                        [AddedToken(piece, normalized=False, special=kind == PieceKind.CONTROL)]
                    )
                    if encoder.token_to_id(piece) != token:
                        raise ValueError("tokenizer construction changed a model token ID")
            self._encoders[handling] = encoder

    @property
    def vocabulary(self) -> int:
        return len(self.config.pieces)

    @property
    def stop_tokens(self) -> frozenset[TokenId]:
        return self.config.stop_tokens

    def encode(
        self, text: str, *, special: SpecialTokens = SpecialTokens.RECOGNIZE
    ) -> tuple[TokenId, ...]:
        if type(text) is not str or not isinstance(special, SpecialTokens):
            raise TypeError("tokenization requires text and a special-token handling policy")
        return tuple(
            TokenId(token)
            for token in self._encoders[special].encode(text, add_special_tokens=False).ids
        )

    def piece(self, token: TokenId, *, skip_control: bool = True) -> bytes:
        if type(token) is not int or not 0 <= token < self.vocabulary:
            raise ValueError("token is outside the vocabulary")
        return (
            b""
            if skip_control and self.config.kinds[token] == PieceKind.CONTROL
            else self._pieces[token]
        )
