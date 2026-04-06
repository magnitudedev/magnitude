// Simple chat BAML function
// System prompt is passed as a parameter (stored in text files, not BAML templates)

function SimpleChat(
  systemPrompt: string,
  messages: ChatMessage[],
  includeSystemPromptMessage: bool
) -> string {
  client ChatClientNoRetryAnthropicOnly
  prompt #"
    {%if includeSystemPromptMessage%}
    {{ _.role("system") }}
    {{ systemPrompt }}
    {%endif%}

    {% for message in messages %}
    {% if message.role == "user" %}
      {{ RenderUserContent(message.content, loop.last) }}
    {% else %}
      {{ RenderAssistantContent(message.content, loop.last) }}
    {% endif %}
    {% endfor %}
  "#
}