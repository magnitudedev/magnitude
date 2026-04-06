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
    <system>
    To demonstrate understanding of the response format, send a message to the recipient "demo" saying "Hello, I understand how to respond" and then idle.
    </system>

    {{ _.role("assistant") }}
    {{ ackTurn }}
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