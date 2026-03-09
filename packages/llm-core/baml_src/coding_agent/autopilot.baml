// Autopilot continuation function
// Uses the same model as the main coding agent to generate
// contextual continuation prompts from the "user" perspective.

function AutopilotContinuation(
  systemPrompt: string,
  messages: ChatMessage[],
  includeClaudeSpoof: bool
) -> string {
  client ChatClientNoRetryAnthropicOnly
  prompt #"
    {%if includeClaudeSpoof%}
    {{ _.role("system") }}
    You are Claude Code, Anthropic's official CLI for Claude.
    {%endif%}
    {{ _.role("system") }}
    {{ systemPrompt }}

    {% for message in messages %}
    {{ _.role(message.role) }}
    {{ RenderContent(message.content) }}
    {% endfor %}

    {{ _.role("assistant") }}
  "#
}