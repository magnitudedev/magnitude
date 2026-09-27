"""One request-level tool selection shared by prompt rendering and output constraints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSelection:
    tools: tuple[dict, ...]
    required: bool
    name: str | None = None

    def instruct(self, messages: list[dict], *, parallel: bool) -> list[dict]:
        """Communicate the selection to the model without changing caller history."""
        if not self.required:
            return messages
        subject = (
            f"the supplied tool {self.name!r}"
            if self.name is not None
            else "at least one of the supplied tools"
            if parallel
            else "one of the supplied tools"
        )
        instruction = f"Call {subject} to answer this request."
        if messages[0]["role"] == "system":
            first = dict(messages[0])
            content = first["content"]
            first["content"] = (
                [*content, {"type": "text", "text": instruction}]
                if isinstance(content, list)
                else f"{content}\n\n{instruction}".strip()
            )
            return [first, *messages[1:]]
        return [{"role": "system", "content": instruction}, *messages]


def select_tools(tools: list[dict], choice: str | dict) -> ToolSelection:
    if isinstance(choice, dict):
        function = choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        selected = tuple(tool for tool in tools if tool["function"]["name"] == name)
        if not selected:
            raise ValueError("tool_choice names an unavailable function")
        return ToolSelection(selected, True, name)
    if choice not in ("auto", "required", "none"):
        raise ValueError("unsupported tool_choice")
    if choice == "required" and not tools:
        raise ValueError("required tool choice needs at least one tool")
    return ToolSelection(() if choice == "none" else tuple(tools), choice == "required")
