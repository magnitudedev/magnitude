// Coding Agent BAML function
// System prompt is passed as a parameter (stored in text files, not BAML templates)

template_string ProseOpen #"<raw>"#
template_string ProseClose #"</raw>"#

function CodingAgentChat(
  systemPrompt: string,
  messages: ChatMessage[],
  ackTurns: ChatMessage[],
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

    {%if ackTurns %}
    {% for msg in ackTurns %}
    {% if msg.role == "user" %}
      {{ RenderUserContent(msg.content, false) }}
    {% else %}
      {{ RenderAssistantContent(msg.content, false) }}
    {% endif %}
    {% endfor %}
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