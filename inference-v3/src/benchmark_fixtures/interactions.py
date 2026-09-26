"""Tool interactions independent of benchmark scheduling and serving."""

from pydantic import JsonValue

from .records import Record, encoded


class ExpectedCall(Record):
    name: str
    arguments: dict[str, list[JsonValue]]


class Interaction(Record):
    id: str
    category: str
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    expected: list[ExpectedCall]
    provenance: dict[str, str]

    def completed(self, identity: str) -> list[dict[str, JsonValue]]:
        calls = []
        replies = []
        for index, call in enumerate(self.expected):
            arguments = {key: values[0] for key, values in call.arguments.items() if values}
            call_id = f"call_{identity}_{index}"
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": encoded(arguments),
                    },
                }
            )
            replies.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": encoded(
                        {
                            "ok": True,
                            "tool": call.name,
                            "arguments": arguments,
                        }
                    ),
                }
            )
        return [*self.messages, {"role": "assistant", "content": "", "tool_calls": calls}, *replies]
