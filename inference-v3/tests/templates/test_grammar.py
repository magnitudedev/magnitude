"""Grammar acceptance must constrain a complete token, including embedded triggers."""

from pathlib import Path

import pytest
from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper

from templates import Template
from templates.grammar import to_lark

FIXTURES = Path(__file__).resolve().parents[2] / "native/templates/upstream/models/templates"
TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class ByteVocabulary:
    """Exact byte vocabulary with optional multi-byte tokens; no model needed."""

    eos_token_id = 256
    bos_token_id = None
    special_token_ids = [256]

    def __init__(self, merged=()):
        self.tokens = [bytes([i]) for i in range(256)] + [b"<eos>", *merged]

    def __call__(self, text):
        return list(text if isinstance(text, bytes) else text.encode())


def matcher(plan, merged=()):
    vocabulary = ByteVocabulary(merged)
    tokenizer = LLTokenizer(
        TokenizerWrapper(vocabulary), n_vocab=len(vocabulary.tokens), eos_token=256
    )
    assert not plan.description.grammar_lazy
    assert not plan.description.grammar_triggers
    grammar = LLMatcher.grammar_from_lark(to_lark(plan.description.grammar))
    result = LLMatcher(tokenizer, grammar, log_level=0)
    assert not result.is_error(), result.get_error()
    assert not result.get_grammar_warnings()
    assert result.consume_tokens(list(plan.description.grammar_initial_prefix.encode()))
    return result


@pytest.mark.parametrize(
    "output,accepted",
    [
        ("ordinary prose 🦙", True),
        ("<think>consider</think>answer", True),
        ('<tool_call>\n{"name":"search","arguments":{"query":"hello"}}\n</tool_call>', True),
        ('<tool_call>\n{"name":"unknown","arguments":{"query":"hello"}}\n</tool_call>', False),
        ('<tool_call>\n{"name":"search","arguments":{"bad":"hello"}}\n</tool_call>', False),
        ('<tool_call>\n{"name":"search","arguments":{"query":42}}\n</tool_call>', False),
        (
            '<tool_call>\n{"name":"search","arguments":{"query":"hello","extra":1}}\n</tool_call>',
            False,
        ),
    ],
)
def test_whole_completion_schema_acceptance(output, accepted):
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "hi"}], tools=[TOOL], now=946684800
        ) as plan:
            state = matcher(plan)
            data = list(output.encode())
            valid_prefix = state.validate_tokens(data)
            if accepted:
                assert valid_prefix == len(data)
                assert state.consume_tokens(data)
                assert state.is_accepting()
            else:
                assert valid_prefix < len(data)


def test_token_crossing_tool_trigger_cannot_hide_invalid_arguments():
    valid = b'prose <tool_call>\n{"name":"search","arguments":{"query":"hello"}}\n</tool_call>'
    invalid = valid.replace(b'"query":"hello"', b'"query":42')
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "hi"}], tools=[TOOL], now=946684800
        ) as plan:
            state = matcher(plan, (valid, invalid))
            assert state.validate_tokens([257]) == 1
            assert state.validate_tokens([258]) == 0
            mask = state.compute_bitmask()
            assert mask[257 // 8] & (1 << (257 % 8))
            assert not mask[258 // 8] & (1 << (258 % 8))
            assert state.consume_token(257)
            assert state.is_accepting()


@pytest.mark.parametrize(
    "filename,output",
    [
        (
            "Qwen3-Coder.jinja",
            "<tool_call>\n<function=search>\n<parameter=query>\nhello\n</parameter>\n</function>\n</tool_call>",
        ),
        (
            "Qwen3.5-4B.jinja",
            "consider</think>\n\n<tool_call>\n<function=search>\n<parameter=query>\nhello\n</parameter>\n</function>\n</tool_call>",
        ),
        (
            "openai-gpt-oss-120b.jinja",
            ' to=functions.search<|channel|>commentary json<|message|>{"query":"hello"}',
        ),
        (
            "deepseek-ai-DeepSeek-V3.1.jinja",
            '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search<｜tool▁sep｜>{"query":"hello"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        ),
        (
            "google-gemma-4-31B-it.jinja",
            '<|tool_call>call:search{query:<|"|>hello<|"|>}<tool_call|>',
        ),
        (
            "mistralai-Mistral-Nemo-Instruct-2407.jinja",
            '[TOOL_CALLS][{"name":"search","arguments":{"query":"hello"},"id":"abc123xyz"}]',
        ),
        (
            "Mistral-Small-3.2-24B-Instruct-2506.jinja",
            '[TOOL_CALLS]search[CALL_ID]abc123xyz[ARGS]{"query":"hello"}',
        ),
    ],
)
def test_specialized_and_automatic_grammars_accept_parser_fixtures(filename, output):
    from copy import deepcopy

    tools = deepcopy([TOOL])
    if filename == "google-gemma-4-31B-it.jinja":
        # Upstream Gemma enforces names but only supports unrestricted dictionaries.
        for tool in tools:
            tool["function"]["parameters"] = {"type": "object"}
    with Template(
        (FIXTURES / filename).read_text(), special_tokens={"bos_token": "<s>", "eos_token": "</s>"}
    ) as template:
        with template.prepare(
            [{"role": "user", "content": "hi"}], tools=tools, now=946684800
        ) as plan:
            state = matcher(plan)
            data = list(output.encode())
            assert state.validate_tokens(data) == len(data)
            assert state.consume_tokens(data)
            assert state.is_accepting()
            plan.parse(output.encode())


@pytest.mark.parametrize("pattern", ["unanchored", "^(?=x)x$"])
def test_unsupported_pattern_conversion_is_an_admission_error(pattern):
    from copy import deepcopy

    from templates import NativeError

    tool = deepcopy(TOOL)
    tool["function"]["parameters"]["properties"]["query"]["pattern"] = pattern
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with pytest.raises(NativeError, match="Unsupported JSON schema conversion"):
            template.prepare([{"role": "user", "content": "hi"}], tools=[tool], now=946684800)


def test_converter_renaming_preserves_literals_and_distinct_normalized_names():
    vocabulary = ByteVocabulary()
    tokenizer = LLTokenizer(TokenizerWrapper(vocabulary), n_vocab=257, eos_token=256)
    grammar = to_lark(
        'root ::= start foo-bar foo_bar\nstart ::= "start"\nfoo-bar ::= "A"\nfoo_bar ::= "B"\n'
    )
    state = LLMatcher(tokenizer, LLMatcher.grammar_from_lark(grammar), log_level=0)
    assert not state.is_error(), state.get_error()
    assert state.consume_tokens(list(b"startAB"))
    assert state.is_accepting()


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "number", "minimum": 1},
        {"type": "array", "uniqueItems": True},
        {"type": "object", "required": ["missing"]},
        {"type": "string", "format": "email"},
        {"oneOf": [{"type": "integer"}, {"type": "number"}]},
        {"anyOf": [{"type": "string"}], "type": "integer"},
        {"type": "string", "enum": [42]},
        {"type": "array", "items": {"type": "string", "minLength": 3}},
        {"$ref": "https://example.invalid/schema"},
    ],
)
def test_unsupported_schema_constraints_fail_before_returning_a_plan(schema):
    from templates import NativeError

    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with pytest.raises(NativeError, match="Unsupported JSON schema"):
            template.prepare([{"role": "user", "content": "hi"}], json_schema=schema, now=946684800)


@pytest.mark.parametrize(
    "schema,valid,invalid",
    [
        (
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            '{"x":"ok","extra":[1,true]}',
            '{"x":42}',
        ),
        (
            {
                "type": "object",
                "properties": {"x": {"enum": ["a", "b"]}},
                "required": ["x"],
                "additionalProperties": False,
            },
            '{"x":"b"}',
            '{"x":"c"}',
        ),
        (
            {"type": "array", "items": {"anyOf": [{"type": "null"}, {"type": "string"}]}},
            '[null,"yes"]',
            "[42]",
        ),
        (
            {
                "$defs": {"value": {"enum": ["a", "b"]}},
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/value"}},
                "required": ["x"],
                "additionalProperties": False,
            },
            '{"x":"a"}',
            '{"x":"c"}',
        ),
    ],
)
def test_response_schema_languages(schema, valid, invalid):
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "hi"}], json_schema=schema, now=946684800
        ) as plan:
            state = matcher(plan)
            valid_tokens = list(valid.encode())
            invalid_tokens = list(invalid.encode())
            assert state.validate_tokens(invalid_tokens) < len(invalid_tokens)
            assert state.consume_tokens(valid_tokens)
            assert state.is_accepting()


@pytest.mark.parametrize(
    "filename", ["Qwen3-Coder.jinja", "Qwen3.5-4B.jinja", "google-gemma-4-31B-it.jinja"]
)
def test_format_specific_unenforced_constraints_are_rejected(filename):
    from copy import deepcopy

    from templates import NativeError

    tool = deepcopy(TOOL)
    tool["function"]["parameters"]["properties"]["query"]["enum"] = ["allowed"]
    with Template((FIXTURES / filename).read_text()) as template:
        with pytest.raises(NativeError, match="Unsupported JSON schema"):
            template.prepare([{"role": "user", "content": "hi"}], tools=[tool], now=946684800)


def test_recursive_scanners_keep_exact_masks_and_delimiter_ambiguity():
    from itertools import product
    from llguidance import gbnf_to_lark as converter
    from llguidance.gbnf_to_lark import gbnf_to_lark
    from templates.regular import orient_regular_regions

    # Both entries reach the same cyclic scanner. The final literal may also
    # match scanner text, so greedy lexer replacement would change this language.
    source = """root ::= "X" scan "ab" | "Y" partial "b"
scan ::= | "a" partial | [bc] scan
partial ::= | "a" partial | "b" scan
"""
    pieces = ByteVocabulary((b"Xaab", b"Yabb", b"Xcab", b"Xabca", b"Yaaaab"))
    tokenizer = LLTokenizer(TokenizerWrapper(pieces), n_vocab=len(pieces.tokens), eos_token=256)
    before = LLMatcher(tokenizer, LLMatcher.grammar_from_lark(gbnf_to_lark(source)))
    rules = converter.GrammarParser().parse(source)
    orient_regular_regions(rules)
    converter.resolve(rules)
    converted = "%llguidance {}\n" + "\n".join(str(rule) for rule in rules.values())
    after = LLMatcher(tokenizer, LLMatcher.grammar_from_lark(converted))
    for prefix in (b"X", b"Y"):
        for length in range(5):
            for suffix in product(b"abc", repeat=length):
                tokens = list(prefix + bytes(suffix))
                left, right = before.deep_copy(), after.deep_copy()
                assert left.validate_tokens(tokens) == right.validate_tokens(tokens)
                accepted = left.consume_tokens(tokens)
                assert right.consume_tokens(tokens) == accepted
                if accepted:
                    assert left.is_accepting() == right.is_accepting()
                    assert left.compute_bitmask() == right.compute_bitmask()
                    assert not left.is_error() and not right.is_error()


def test_long_prose_before_tool_does_not_accumulate_scanner_endings():
    with Template((FIXTURES / "Qwen-Qwen3-0.6B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "hi"}], tools=[TOOL], tool_choice="required", now=946684800
        ) as plan:
            state = matcher(plan)
            assert state.consume_tokens(list(b"a " * 8192))
            state.compute_bitmask()
            assert not state.is_error(), state.get_error()
            call = b'<tool_call>\n{"name":"search","arguments":{"query":"hello"}}\n</tool_call>'
            assert state.consume_tokens(list(call))
            assert state.is_accepting()


def test_whole_regular_language_matches_independent_acceptance_and_prefix_oracle():
    import re
    from itertools import product

    source = """root ::= "X" scan "ab" | "Y" partial "b"
scan ::= | "a" partial | [bc] scan
partial ::= | "a" partial | "b" scan
"""
    # Direct solution of the two-state automaton, independent of the converter.
    language = re.compile(rb"(?:X(?:[bc]|a+b)*a*ab|Ya*(?:b(?:[bc]|a+b)*a*)?b)")
    pieces = ByteVocabulary((b"Xaab", b"Yabb", b"Xcab", b"Xabca", b"Yaaaab"))
    tokenizer = LLTokenizer(TokenizerWrapper(pieces), n_vocab=len(pieces.tokens), eos_token=256)
    converted = to_lark(source)
    assert "start: WHOLE_COMPLETION" in converted
    initial = LLMatcher(tokenizer, LLMatcher.grammar_from_lark(converted))
    endings = [bytes(chars) for length in range(3) for chars in product(b"abc", repeat=length)]
    for first in (b"X", b"Y"):
        for length in range(5):
            for suffix in product(b"abc", repeat=length):
                prefix = first + bytes(suffix)
                state = initial.deep_copy()
                live = state.consume_tokens(list(prefix))
                assert (live and state.is_accepting()) == bool(language.fullmatch(prefix))
                if not live:
                    assert not any(language.fullmatch(prefix + end) for end in endings)
                    continue
                mask = state.compute_bitmask()
                assert not state.is_error()
                for token, value in enumerate(pieces.tokens):
                    expected = (
                        bool(language.fullmatch(prefix))
                        if token == 256
                        else any(language.fullmatch(prefix + value + end) for end in endings)
                    )
                    assert bool(mask[token // 8] & (1 << (token % 8))) == expected, (prefix, value)


def test_required_tool_prefix_tokens_do_not_inherit_greedy_cfg_lexer_boundaries():
    with Template((FIXTURES / "Qwen3.5-4B.jinja").read_text()) as template:
        with template.prepare(
            [{"role": "user", "content": "hi"}],
            tools=[TOOL],
            tool_choice="required",
            template_arguments={"enable_thinking": False},
            now=946684800,
        ) as plan:
            prefixes = (b"<td", b"\n\n\n\n", b"<think>")
            call = b"<tool_call>\n<function=search>\n<parameter=query>\nhello\n</parameter>\n</function>\n</tool_call>"
            for index, prefix in enumerate(prefixes):
                state = matcher(plan, prefixes)
                mask = state.compute_bitmask()
                token = 257 + index
                assert mask[token // 8] & (1 << (token % 8))
                assert state.consume_tokens([token, *call])
                assert state.is_accepting()
                plan.parse(prefix + call)


@pytest.mark.parametrize(
    "grammar,accepted,rejected",
    [
        (r'root ::= "\u0000\u000a\U0001f600"', "\x00\n😀", "\x00\n😁"),
        (r'root ::= "\x22\\\t\r\""', '"\\\t\r"', '"\\\t\n"'),
        (r"root ::= [\u0041-\u0043]+", "ABC", "ABD"),
        (r"root ::= [\U0001f600-\U0001f602]+", "😀😁😂", "😀😃"),
        (r"root ::= [\x00-\U0010FFFF]*", "\x00\n世界😀", None),
    ],
)
def test_unicode_escape_width_preserves_literal_and_range_languages(grammar, accepted, rejected):
    vocabulary = ByteVocabulary()
    tokenizer = LLTokenizer(
        TokenizerWrapper(vocabulary), n_vocab=len(vocabulary.tokens), eos_token=256
    )
    compiled = LLMatcher.grammar_from_lark(to_lark(grammar))
    for text, expected in ((accepted, True), (rejected, False)):
        if text is None:
            continue
        state = LLMatcher(tokenizer, compiled, log_level=0)
        assert not state.is_error(), state.get_error()
        assert not state.get_grammar_warnings()
        consumed = state.consume_tokens(list(text.encode()))
        assert (consumed and state.is_accepting()) is expected
