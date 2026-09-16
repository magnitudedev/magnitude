import pytest

from engine.data import TokenId
from engine.generation.constraints import ConstraintVocabulary
from engine.inputs.tokenizer import BPEConfig, PieceKind, SpecialTokens
from engine.weights.identity import ArtifactIdentity


def vocabulary(*, normalize=False):
    included = {*range(33, 127), *range(161, 173), *range(174, 256)}
    missing = [value for value in range(256) if value not in included]
    pieces = tuple(
        chr(value) if value in included else chr(256 + missing.index(value)) for value in range(256)
    )
    config = BPEConfig(
        artifact_identity=ArtifactIdentity("0" * 64),
        pieces=(*pieces, "<end>", "<stop>", "<control>", "<added>", "<unused>"),
        kinds=(
            *((PieceKind.NORMAL,) * 256),
            PieceKind.CONTROL,
            PieceKind.CONTROL,
            PieceKind.CONTROL,
            PieceKind.USER_DEFINED,
            PieceKind.UNUSED,
        ),
        merges=(),
        pattern=r".+|\s",
        normalize_nfc=normalize,
        stop_tokens=frozenset({TokenId(256), TokenId(257)}),
    )
    return ConstraintVocabulary(config, projection_vocabulary=288)


def allowed(mask, token):
    return bool(mask[token // 8] & (1 << (token % 8)))


def test_binding_preserves_bytes_special_tokens_and_encoding_policy():
    binding = vocabulary()
    for token in (*range(255), *range(256, 260)):
        assert binding.native.decode_bytes([token]) == binding.tokenizer.piece(
            TokenId(token), skip_control=False
        )
    for text in ("héllo 世界 🦙", "x\x00y", "<control><added>", "e\u0301"):
        expected = binding.tokenizer.encode(text, special=SpecialTokens.RECOGNIZE)
        assert binding.native.tokenize_str(text) == list(expected)
    assert binding.native.eos_tokens == [256, 257]


def test_mask_tracks_partial_utf8_and_both_eos_without_padding_or_unused_ids():
    binding = vocabulary()
    matcher = binding.matcher('root ::= "🦙"')
    assert len(binding.allowed_mask(matcher)) == ((binding.projection_vocabulary + 31) // 32) * 4
    for byte in "🦙".encode():
        mask = binding.allowed_mask(matcher)
        assert allowed(mask, byte)
        assert not allowed(mask, 256) and not allowed(mask, 257)
        assert all(not allowed(mask, token) for token in (255, *range(260, len(mask) * 8)))
        assert matcher.consume_token(byte)
    assert matcher.is_accepting()
    mask = binding.allowed_mask(matcher)
    assert allowed(mask, 256) and allowed(mask, 257)
    for eos in (256, 257):
        copy = matcher.deep_copy()
        assert copy.consume_token(eos)
        assert copy.is_stopped()


def test_request_matchers_are_independent_and_prefix_is_not_the_conversation():
    binding = vocabulary()
    first = binding.matcher('root ::= "prefix" "yes"', initial_prefix="prefix")
    second = binding.matcher('root ::= "prefix" "yes"', initial_prefix="prefix")
    assert first.consume_tokens(list(b"yes"))
    assert first.is_accepting()
    assert not second.is_accepting()
    assert second.validate_tokens(list(b"yes")) == 3
    with pytest.raises(ValueError, match="generation prefix"):
        binding.matcher('root ::= "prefix" "yes"', initial_prefix="conversation prefix")


def test_normalizing_tokenizer_cannot_silently_change_grammar_prefix():
    with pytest.raises(ValueError, match="exactly representable"):
        vocabulary(normalize=True).matcher('root ::= "é"', initial_prefix="e\u0301")


def test_control_delimiters_match_the_same_visible_grammar_bytes():
    binding = vocabulary()
    matcher = binding.matcher('root ::= "<control>" "ok"', initial_prefix="<control>")
    assert matcher.consume_tokens(list(b"ok"))
    assert matcher.is_accepting()
    matcher = binding.matcher('root ::= "<control>" "ok"')
    assert matcher.validate_tokens([258, 111, 107]) == 3
    assert matcher.validate_tokens(list(b"<control>ok")) == len(b"<control>ok")


def test_prepared_native_plan_uses_engine_tokenization_end_to_end():
    from pathlib import Path

    from templates import Template

    source = (
        Path(__file__).resolve().parents[2]
        / "native/templates/upstream/models/templates/Qwen-Qwen3-0.6B.jinja"
    ).read_text()
    binding = vocabulary()
    with Template(source) as template:
        with template.prepare(
            [{"role": "user", "content": "hello"}],
            json_schema={
                "type": "object",
                "properties": {"answer": {"enum": ["🦙"]}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            now=946684800,
        ) as plan:
            matcher = binding.matcher(
                plan.description.grammar, initial_prefix=plan.description.grammar_initial_prefix
            )
            output = '{"answer":"🦙"}'
            for token in binding.tokenizer.encode(output):
                assert allowed(binding.allowed_mask(matcher), token)
                assert matcher.consume_token(token)
            assert matcher.is_accepting()
            assert allowed(binding.allowed_mask(matcher), 256)
            assert allowed(binding.allowed_mask(matcher), 257)
            plan.parse(output.encode())


def test_constraint_transitions_are_speculative_and_fail_without_partial_acceptance():
    from engine.generation.constraints import ConstraintState

    state = ConstraintState(vocabulary(), 'root ::= "yes"')
    before = state.mask()
    with pytest.raises(ValueError, match="violate"):
        state.stage(tuple(map(TokenId, b"yet")))
    assert state.position == 0 and state.mask() == before
    discarded = state.stage((TokenId(ord("y")),))
    assert state.position == 0 and state.mask() == before
    chosen = state.stage(tuple(map(TokenId, b"yes")))
    chosen.commit()
    assert state.position == 3 and state.accepting
    with pytest.raises(RuntimeError, match="stale"):
        discarded.commit()
    with pytest.raises(RuntimeError, match="committed"):
        chosen.commit()
    assert state.position == 3


def test_constraint_fork_and_forced_proposals_do_not_consume_or_share_state():
    from engine.generation.constraints import ConstraintState

    state = ConstraintState(vocabulary(), 'root ::= "hello"')
    initial = state.mask()
    assert state.forced(2) == tuple(b"he")
    assert state.forced(20) == tuple(b"hello")
    assert state.position == 0 and state.mask() == initial
    state.stage(state.forced(2)).commit()
    fork = state.fork()
    fork.stage(fork.forced(20)).commit()
    assert fork.position == 5 and fork.accepting
    assert state.position == 2 and not state.accepting
    fork.stage((TokenId(257),)).commit()
    assert fork.stopped and not state.stopped
    assert fork.forced(20) == ()


@pytest.mark.parametrize("tokens", [(255,), (260,), (280,), (-1,), (True,), (256, 121)])
def test_unusable_and_premature_eos_transitions_leave_state_unchanged(tokens):
    from engine.generation.constraints import ConstraintState

    state = ConstraintState(vocabulary(), 'root ::= "yes"')
    before = state.mask()
    with pytest.raises(ValueError):
        state.stage(tokens)
    assert state.position == 0 and state.mask() == before


def test_compiled_cache_is_bounded_and_never_shares_request_progress():
    original = vocabulary()
    binding = ConstraintVocabulary(
        original.tokenizer.config, projection_vocabulary=288, cache_entries=2, cache_bytes=4096
    )
    first = binding.matcher('root ::= "hello"')
    assert first.consume_tokens(list(b"he"))
    second = binding.matcher('root ::= "hello"')
    assert binding.cache_hits == 1 and binding.cache_misses == 1
    assert second.consume_tokens(list(b"hello")) and second.is_accepting()
    assert not first.is_accepting()
    binding.matcher('root ::= "yes"')
    binding.matcher('root ::= "no"')
    binding.matcher('root ::= "hello"')
    assert binding.cache_misses == 4 and len(binding._compiled) == 2
    assert 0 < binding.cache_bytes <= 4096
    # Different prompt prefixes cannot reuse a matcher initialized further ahead.
    initialized = binding.matcher('root ::= "hello"', initial_prefix="he")
    assert initialized.consume_tokens(list(b"llo")) and initialized.is_accepting()
    assert binding.cache_misses == 5
    before = (len(binding._compiled), binding.cache_bytes)
    with pytest.raises(ValueError):
        binding.matcher('root ::= "x"', initial_prefix="bad")
    assert (len(binding._compiled), binding.cache_bytes) == before


def test_initial_mask_is_prepared_before_admission_and_cache_is_invalidated_only_on_commit():
    from engine.generation.constraints import ConstraintState

    binding = vocabulary()
    first = ConstraintState(binding, 'root ::= "ab"')
    second = ConstraintState(binding, 'root ::= "ab"')
    assert first.compilation.initial_mask_ns > 0
    assert second.compilation.cache_hit and second.compilation.initial_mask_ns == 0
    initial = first.mask()
    assert second.mask() == initial
    transition = first.stage((TokenId(ord("a")),))
    assert first.mask() is initial
    transition.commit()
    following = first.mask()
    assert allowed(following, ord("b")) and not allowed(following, ord("a"))
    assert first.mask() is following and second.mask() == initial


@pytest.mark.parametrize("dense", [False, True])
def test_mask_sanitization_preserves_every_usable_bit_and_clears_all_other_ids(dense):
    config = vocabulary().tokenizer.config
    if dense:
        kinds = tuple(
            PieceKind.UNUSED if i < 256 and i % 8 == 0 else kind
            for i, kind in enumerate(config.kinds)
        )
        config = config.model_copy(update={"kinds": kinds})
    binding = ConstraintVocabulary(config, projection_vocabulary=320)
    # A deliberately unsafe mask exercises the boundary independent of the
    # matcher's current habit of leaving unused token bits clear itself.
    source = bytes((i * 37 + 17) % 256 for i in range(44))

    class Mask:
        def is_error(self):
            return False

        def compute_bitmask(self):
            return source

    filtered = binding.allowed_mask(Mask())
    assert len(filtered) == 40
    for token in range(320):
        usable = token < len(config.pieces) and token not in binding.unused
        assert allowed(filtered, token) == (usable and allowed(source, token))
