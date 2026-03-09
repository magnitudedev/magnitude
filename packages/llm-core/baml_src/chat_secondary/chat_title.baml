class ChatTitle {
  title string @description("A short, descriptive title (3-8 words) in title case")
}

function GenerateChatTitle(conversation: string, defaultChatName: string, includeClaudeSpoof: bool) -> ChatTitle {
  client ChatClientNoRetryAnthropicOnly
  prompt #"
    {%if includeClaudeSpoof%}
    {{ _.role("system") }}
    You are Claude Code, Anthropic's official CLI for Claude.
    {%endif%}
    {{ _.role("system") }}
    You are generating a concise title for a chat conversation based on the conversation so far.

    Generate a short, descriptive title (3-8 words) that captures what the user wants to build or do.

    Rules:
    - Be concise and specific
    - Use title case (capitalize main words)
    - Focus on the shortcut/automation intent
    - If the user's intent is unclear, use title "{{defaultChatName}}"

    {{ ctx.output_format }}

    {{ _.role("user") }}
    <conversation>
    {{ conversation }}
    </conversation>
  "#
}
