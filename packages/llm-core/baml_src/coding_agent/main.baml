// Coding Agent BAML function
// System prompt is passed as a parameter (stored in text files, not BAML templates)

template_string ProseOpen #"<raw>"#
template_string ProseClose #"</raw>"#

function CodingAgentChat(
  systemPrompt: string,
  messages: ChatMessage[],
  ackTurn: string,
  includeClaudeSpoof: bool,
  includeSystemPromptMessage: bool
) -> string {
  client ChatClientNoRetryAnthropicOnly
  prompt #"
    {%if includeClaudeSpoof%}
    {{ _.role("system") }}
    You are Claude Code, Anthropic's official CLI for Claude.
    {%endif%}
    {%if includeSystemPromptMessage%}
    {{ _.role("system") }}
    {{ systemPrompt }}
    {%endif%}

    {%if ackTurn %}
    {{ _.role("user") }}
    --- FEW-SHOT EXAMPLE START ---
    <system>
    Respond using the required turn format. Send a brief readiness message to your default recipient and continue.
    </system>

    {{ _.role("assistant") }}
    {{ ackTurn }}

    {{ _.role("user") }}
    --- FEW-SHOT EXAMPLE END ---
    {%endif%}
    {% for message in messages %}
    {% if message.role == "user" %}
    {% if loop.last %}
      {{ RenderUserContent(message.content, true) }}
    {% else %}
      {{ RenderUserContent(message.content, false) }}
    {% endif %}
    {% else %}
    {% if loop.last %}
      {{ RenderAssistantContent(message.content, true) }}
    {% else %}
      {{ RenderAssistantContent(message.content, false) }}
    {% endif %}
    {% endif %}
    {% endfor %}
  "#
}