"""Request languages composed from tool schemas and the checkpoint's argument wire."""

import json

from magnitude_engine.generation.constraint_spec import ConstraintSpec

from .formats import ChatFormat
from .tool_choice import select_tools


def literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def json_language(schema: dict) -> str:
    return "%json " + json.dumps({**schema, "x-guidance": {"whitespace_flexible": True}})


class ArgumentGrammar:
    def __init__(self):
        self.rules = ["space: /[ \\t\\r\\n]*/", 'raw[suffix="</parameter>"]: /(.|\\n)*/']

    def rule(self, expression: str) -> str:
        name = f"argument_{len(self.rules)}"
        self.rules.append(f"{name}: {expression}")
        return name

    def schema(self, schema: dict, root: dict) -> dict:
        # Resolve local definition aliases, leaving recursive JSON schemas to llguidance.
        seen = set()
        while set(schema) == {"$ref"}:
            ref = schema["$ref"]
            if ref in seen:
                raise ValueError("recursive aliases cannot describe a tool argument wire")
            seen.add(ref)
            if not ref.startswith("#/"):
                raise ValueError("tool schemas require local references")
            value = root
            for key in ref[2:].split("/"):
                value = value[key.replace("~1", "/").replace("~0", "~")]
            schema = value
        return schema

    def xml_value(self, schema: dict, root: dict) -> str:
        schema = self.schema(schema, root)
        kind = schema.get("type")
        if isinstance(kind, list):
            kind = kind[0] if len(kind) == 1 else None
        if schema.get("enum"):
            values = [v if isinstance(v, str) else json.dumps(v) for v in schema["enum"]]
            value = "(" + " | ".join(literal(v) for v in values) + ")"
        elif kind in (None, "string"):
            return "raw"
        elif kind == "boolean":
            value = '("true" | "false" | "True" | "False")'
        elif kind == "null":
            value = '("null" | "None")'
        else:
            value = self.rule(
                json_language(
                    {
                        **{key: root[key] for key in ("$defs", "definitions") if key in root},
                        **schema,
                    }
                )
            )
        return f'{value} space "</parameter>"'

    def xml(self, function: dict) -> str:
        root = function.get("parameters") or {}
        parts = [literal("<function=" + function["name"] + ">"), "space"]
        for name, schema in root.get("properties", {}).items():
            block = (
                f"{literal('<parameter=' + name + '>')} space {self.xml_value(schema, root)} space"
            )
            parts.append(block if name in root.get("required", ()) else f"({block})?")
        parts.extend(['"</function>"', "space"])
        return self.rule(" ".join(parts))

    def gemma_value(self, schema: dict, root: dict) -> str:
        schema = self.schema(schema, root)
        if schema.get("enum"):
            return (
                "("
                + " | ".join(
                    f'<|"|> {literal(v)} <|"|>' if isinstance(v, str) else literal(json.dumps(v))
                    for v in schema["enum"]
                )
                + ")"
            )
        kind = schema.get("type")
        annotations = {
            "type",
            "description",
            "title",
            "default",
            "examples",
            "$defs",
            "definitions",
        }
        structural = (
            {"properties", "required", "additionalProperties"} if kind == "object" else {"items"}
        )
        if kind in ("object", "array") and set(schema) - annotations - structural:
            # Rich nested schemas retain llguidance's JSON language, also accepted by this
            # wire's decoder; native Gemma serialization must not erase schema restrictions.
            return self.rule(
                json_language(
                    {
                        **{key: root[key] for key in ("$defs", "definitions") if key in root},
                        **schema,
                    }
                )
            )
        if kind == "object" or "properties" in schema:
            return self.gemma_object(schema, root)
        if kind == "array":
            item = self.gemma_value(schema.get("items", {}), root)
            return self.rule(f'"[" space ({item} (space "," space {item})*)? space "]"')
        if kind in (None, "string"):
            return "gstring"
        return self.rule(json_language(schema))

    def gemma_object(self, schema: dict, root: dict) -> str:
        # Two suffix states avoid exponential expansion of optional fields and prevent
        # leading/trailing commas. Field order follows the checkpoint's dictsort template.
        absent, present = self.rule('""'), self.rule('""')
        for name in reversed(sorted(schema.get("properties", {}), key=str.lower)):
            value = self.gemma_value(schema["properties"][name], root)
            pair = f'{literal(name)} space ":" space {value} space'
            required = name in schema.get("required", ())
            absent_next = self.rule(f"{pair} {present}" + ("" if required else f" | {absent}"))
            present_next = self.rule(
                f'"," space {pair} {present}' + ("" if required else f" | {present}")
            )
            absent, present = absent_next, present_next
        return self.rule(f'"{{" space {absent} "}}"')

    def calls(self, format: ChatFormat, tools: list[dict]) -> str:
        if format.arguments == "json":
            alternatives = [
                {
                    "type": "object",
                    "properties": {
                        "name": {"const": t["function"]["name"]},
                        "arguments": t["function"].get("parameters", {"type": "object"}),
                    },
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                }
                for t in tools
            ]
            return self.rule(json_language({"anyOf": alternatives}))
        if format.arguments == "xml":
            return self.rule(" | ".join(self.xml(t["function"]) for t in tools))
        self.rules.extend(
            [
                'GEMMA_TEXT: /(.|\\n)*/ & ~/(?s:.*)<\\|"\\|>(?s:.*)/',
                'gstring: <|"|> GEMMA_TEXT <|"|>',
            ]
        )
        alternatives = []
        for tool in tools:
            function = tool["function"]
            parameters = function.get("parameters") or {}
            arguments = self.gemma_object(parameters, parameters)
            alternatives.append(f"{literal('call:' + function['name'])} space {arguments}")
        return self.rule(" | ".join(alternatives))


def chat_constraint(
    format: ChatFormat | None,
    available_markers: frozenset[str],
    tools: list[dict],
    *,
    choice: str | dict = "auto",
    parallel: bool = True,
    reasoning_prefilled: bool = False,
    response_format: dict | None = None,
) -> ConstraintSpec | None:
    selection = select_tools(tools, choice)
    tools = list(selection.tools)
    response_format = response_format or {"type": "text"}
    if response_format.get("type") not in ("text", "json_object", "json_schema"):
        raise ValueError("unsupported response_format")
    if not tools and response_format["type"] == "text":
        return None
    if tools and response_format["type"] != "text":
        raise ValueError("tool and response-format constraints cannot be combined")
    builder = ArgumentGrammar()
    builder.rules.append("text: /(.|\\n)*/")
    reasoning = (
        format is not None and {format.reasoning_open, format.reasoning_close} <= available_markers
    )
    if reasoning:
        assert format is not None
        builder.rules.append(f"reason: {format.reasoning_open} text {format.reasoning_close}")
    if tools:
        if format is None or not {format.call_open, format.call_close} <= available_markers:
            raise ValueError("checkpoint cannot enforce its tool-call markers")
        builder.rules.append("free: text (reason text)*" if reasoning else "free: text")
        lead = (
            f"(text {format.reasoning_close})? free"
            if reasoning and reasoning_prefilled
            else "free"
        )
        body = builder.calls(format, tools)
        builder.rules.append(f"call: {format.call_open} space {body} space {format.call_close}")
        required = selection.required
        repeats = "+" if required and parallel else "*" if parallel else "" if required else "?"
        start = f"{lead} (call free){repeats}"
    else:
        schema = (
            {"type": "object"}
            if response_format["type"] == "json_object"
            else response_format.get("json_schema", {}).get("schema")
        )
        if not isinstance(schema, dict):
            raise ValueError("json_schema response format requires a schema object")
        lead = "(reason)?" if reasoning else ""
        if reasoning and reasoning_prefilled:
            assert format is not None
            lead = f"(text {format.reasoning_close})? {lead}"
        start = f"{lead} space {builder.rule(json_language(schema))} space"
    return ConstraintSpec("%llguidance {}\n" + "\n".join(builder.rules) + f"\nstart: {start}\n")
