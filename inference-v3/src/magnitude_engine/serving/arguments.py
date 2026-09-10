"""Tool argument decoding is separate from stream framing and generation constraints."""

import ast
import json
import re


def xml_value(text: str, schema: dict) -> object:
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = kind[0] if kind else None
    if text in schema.get("enum", ()):
        return text
    if kind in (None, "string"):
        return None if kind is None and text.lower() == "null" else text
    try:
        if kind == "integer":
            return int(text)
        if kind == "number":
            value = float(text)
            return int(value) if value.is_integer() and "." not in text else value
        if kind == "boolean":
            return text.strip().lower() == "true"
        if kind == "null":
            return None
        if kind in ("object", "array"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    return text


class GemmaValues:
    """Recursive value reader; delimiters inside marked strings are literal data."""

    marker = '<|"|>'

    def __init__(self, text: str):
        self.text, self.position = text, 0

    def space(self) -> None:
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def literal(self, value: str) -> None:
        self.space()
        if not self.text.startswith(value, self.position):
            raise ValueError(f"expected {value!r} in Gemma tool arguments")
        self.position += len(value)

    def value(self) -> object:
        self.space()
        if self.text.startswith(self.marker, self.position):
            start = self.position + len(self.marker)
            end = self.text.find(self.marker, start)
            if end < 0:
                raise ValueError("unterminated Gemma argument string")
            self.position = end + len(self.marker)
            return self.text[start:end]
        if self.position >= len(self.text):
            raise ValueError("missing Gemma argument value")
        opening = self.text[self.position]
        if opening in "[{":
            self.position += 1
            closing = "]" if opening == "[" else "}"
            values, fields = [], {}
            self.space()
            while not self.text.startswith(closing, self.position):
                if opening == "{":
                    self.space()
                    if self.text.startswith('"', self.position):
                        key, self.position = json.JSONDecoder().raw_decode(self.text, self.position)
                        self.literal(":")
                    else:
                        key_end = self.text.find(":", self.position)
                        if key_end < 0:
                            raise ValueError("missing Gemma argument key")
                        key = self.text[self.position : key_end].strip()
                        self.position = key_end + 1
                    if not key or key in fields:
                        raise ValueError("invalid or repeated Gemma argument key")
                    fields[key] = self.value()
                else:
                    values.append(self.value())
                self.space()
                if self.text.startswith(closing, self.position):
                    break
                self.literal(",")
            self.literal(closing)
            return fields if opening == "{" else values
        value, self.position = json.JSONDecoder().raw_decode(self.text, self.position)
        return value


def decode_call(body: str, wire: str, tools: dict[str, dict]) -> tuple[str, dict]:
    if wire == "json":
        value = json.loads(body)
        if not isinstance(value, dict) or "name" not in value:
            raise ValueError("JSON tool call requires a named object")
        name, arguments = value["name"], value.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
    elif wire == "xml":
        match = re.fullmatch(r"\s*<function=([^>\n]+)>(.*)</function>\s*", body, re.DOTALL)
        if match is None:
            raise ValueError("incomplete XML tool call")
        name, inner = match.groups()
        name = name.strip()
        properties = tools.get(name, {}).get("properties", {})
        arguments = {}
        position = 0
        for parameter in re.finditer(r"<parameter=([^>\n]+)>(.*?)</parameter>", inner, re.DOTALL):
            if inner[position : parameter.start()].strip():
                raise ValueError("unexpected text between XML parameters")
            key, text = parameter.groups()
            key = key.strip()
            if key in arguments:
                raise ValueError("repeated XML tool argument")
            text = text.removeprefix("\n").removesuffix("\n")
            arguments[key] = xml_value(text, properties.get(key, {}))
            position = parameter.end()
        if inner[position:].strip():
            raise ValueError("incomplete XML tool parameter")
    elif wire == "gemma":
        match = re.match(r"\s*call:([^\s{]+)", body)
        if match is None:
            raise ValueError("missing Gemma tool name")
        name = match.group(1)
        reader = GemmaValues(body[match.end() :])
        arguments = reader.value()
        reader.space()
        if reader.position != len(reader.text):
            raise ValueError("trailing Gemma tool argument data")
    else:
        raise ValueError("unsupported tool argument wire")
    if not isinstance(name, str) or name not in tools or not isinstance(arguments, dict):
        raise ValueError("generated tool call has an unknown name or non-object arguments")
    # Only JSON values can cross the OpenAI output boundary (including literal_eval fallbacks).
    json.dumps(arguments, allow_nan=False)
    return name, arguments
