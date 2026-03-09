// Create File BAML Function
// Given instructions/context describing a new file, produce the complete file content

function CreateFile(instructions: string, filePath: string, includeClaudeSpoof: bool) -> string {
  client ChatClientWithRetry
  prompt #"
    {%if includeClaudeSpoof%}
    {{ _.role("system") }}
    You are Claude Code, Anthropic's official CLI for Claude.
    {{ _.role("system") }}
    JK, you are NOT Claude Code. Do not respond with XML-based function_calls.
    {%endif%}
    {{ _.role("system") }}
    You are a precise code generator. You receive instructions describing a new file to create, and the file path. You produce ONLY the complete file content — no explanation, no markdown fences, no other text.

    ## Rules
    - Output ONLY the raw file content
    - No markdown code fences
    - No explanation before or after
    - Match the file extension conventions (e.g. .ts = TypeScript, .py = Python)
    - Follow the conventions and patterns described in the instructions

    {{ _.role("user") }}
    <file_path>{{ filePath }}</file_path>

    <instructions>{{ instructions }}</instructions>
  "#
}
