import llguidance as llg
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from magnitude_engine.generation.constraint_spec import ConstraintSpec
from magnitude_engine.generation.guidance import GuidanceCompiler
from magnitude_engine.serving.formats import format_for
from magnitude_engine.serving.grammar import chat_constraint


@pytest.fixture(scope="module")
def languages():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    markers = {"[EOS]", '<|"|>'}
    for family in ("qwen3", "qwen3_5", "gemma4"):
        wire = format_for(family)
        markers.update((wire.call_open, wire.call_close, wire.reasoning_open, wire.reasoning_close))
    tokenizer.add_special_tokens(sorted(markers))
    eos = tokenizer.token_to_id("[EOS]")
    table = llg.LLTokenizer(tokenizer.to_str(), eos_token=eos)
    return GuidanceCompiler(lambda: table), table, frozenset(markers)


def accepts(languages, grammar, text):
    compiler, table, _ = languages
    matcher = compiler.create(grammar)
    result = (
        matcher.consume(table.eos_token)
        if not text
        else all(matcher.consume(token) for token in table.tokenize_str(text, parse_special=True))
        and matcher.consume(table.eos_token)
    )
    matcher.close()
    return result


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer"},
                    "exact": {"type": "boolean"},
                },
                "required": ["query", "count"],
                "additionalProperties": False,
            },
        },
    }
]


@pytest.mark.parametrize(
    "family,body",
    [
        ("qwen3", '{"name":"search","arguments":{"query":"café","count":2}}'),
        (
            "qwen3_5",
            "<function=search>\n<parameter=query>\ncafé\n</parameter>\n"
            "<parameter=count>2</parameter><parameter=exact>True</parameter></function>",
        ),
        ("gemma4", 'call:search{count:2,exact:true,query:<|"|>café<|"|>}'),
    ],
)
def test_required_optional_parallel_and_reasoning_languages(languages, family, body):
    wire = format_for(family)
    call = wire.call_open + body + wire.call_close
    arguments = (wire, languages[2], TOOLS)
    required = chat_constraint(*arguments, choice="required", parallel=False)
    assert accepts(languages, required, call)
    assert not accepts(languages, required, "")
    assert not accepts(languages, required, "just text")
    assert not accepts(languages, required, call + call)
    assert accepts(
        languages,
        required,
        wire.reasoning_open + wire.reasoning_label + "plan" + wire.reasoning_close + "\n\n" + call,
    )
    assert not accepts(languages, required, call.replace("search", "unknown"))
    parallel = chat_constraint(*arguments, choice="required", parallel=True)
    assert accepts(languages, parallel, call + "\n" + call)
    optional = chat_constraint(*arguments)
    assert accepts(languages, optional, "") and accepts(languages, optional, "text")
    prefilled = chat_constraint(*arguments, choice="required", reasoning_prefilled=True)
    assert accepts(languages, prefilled, "plan" + wire.reasoning_close + "\n" + call)


def test_nested_gemma_wire_and_optional_field_comma_states(languages):
    wire = format_for("gemma4")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["x"],
                        },
                        "c": {"type": "boolean"},
                    },
                    "required": ["b"],
                },
            },
        }
    ]
    spec = chat_constraint(wire, languages[2], tools, choice="required")
    for fields in ("b:{x:[]}", 'a:2,b:{x:[<|"|>a,b{}<|"|>]},c:false', "b:{x:[]},c:true"):
        assert accepts(languages, spec, "<|tool_call>call:f{" + fields + "}<tool_call|>")
    for fields in ("a:2", "b:{}", ",b:{x:[]}", "b:{x:[]},", "b:{x:[false]}"):
        assert not accepts(languages, spec, "<|tool_call>call:f{" + fields + "}<tool_call|>")


def test_schema_output_and_named_tool_selection_are_enforced(languages):
    wire = format_for("qwen3")
    schema = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            }
        },
    }
    spec = chat_constraint(wire, languages[2], [], response_format=schema)
    assert accepts(languages, spec, '<think>plan</think> {"x":3}')
    for text in ('{"x":"3"}', "{}", '{"x":3,"y":4}', "[]"):
        assert not accepts(languages, spec, text)
    chosen = chat_constraint(
        wire, languages[2], TOOLS, choice={"type": "function", "function": {"name": "search"}}
    )
    assert chosen is not None and not accepts(languages, chosen, "text")
    with pytest.raises(ValueError, match="unavailable"):
        chat_constraint(wire, languages[2], TOOLS, choice={"function": {"name": "missing"}})
    with pytest.raises(ValueError, match="markers"):
        chat_constraint(wire, frozenset(), TOOLS, choice="required")
    assert chat_constraint(wire, languages[2], TOOLS, choice="none") is None


def test_reviewed_tool_language_decisions_match_the_external_reference(languages, poc_module):
    grammar = poc_module("chat.grammar")
    records = poc_module("chat.records")
    compiler, table, markers = languages
    for family, body in (
        ("qwen3", '{"name":"search","arguments":{"query":"x","count":2}}'),
        (
            "qwen3_5",
            "<function=search><parameter=query>x</parameter>"
            "<parameter=count>2</parameter></function>",
        ),
        ("gemma4", 'call:search{count:2,query:<|"|>x<|"|>}'),
    ):
        record = records.record_for(family)
        facts = {marker: table.tokenize_str(marker, parse_special=True)[0] for marker in markers}
        wire = format_for(family)
        call = wire.call_open + body + wire.call_close
        for choice, parallel in (("auto", False), ("required", False), ("required", True)):
            ours = chat_constraint(wire, markers, TOOLS, choice=choice, parallel=parallel)
            reference = grammar.build_tool_grammar(
                record, facts, TOOLS, tool_choice=choice, parallel=parallel
            )
            expected = ConstraintSpec(reference.lark)
            for output in (
                "",
                "plain text",
                call,
                call + "\n" + call,
                call.replace("search", "unknown"),
                wire.reasoning_open + wire.reasoning_label + "plan" + wire.reasoning_close + call,
            ):
                assert accepts(languages, ours, output) == accepts(languages, expected, output)


def test_rich_nested_gemma_schemas_keep_array_bounds(languages):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "array", "minItems": 2, "items": {"type": "integer"}}
                    },
                    "required": ["x"],
                },
            },
        }
    ]
    spec = chat_constraint(format_for("gemma4"), languages[2], tools, choice="required")
    assert accepts(languages, spec, "<|tool_call>call:f{x:[1,2]}<tool_call|>")
    assert not accepts(languages, spec, "<|tool_call>call:f{x:[1]}<tool_call|>")
