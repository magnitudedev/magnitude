// Compaction function for the Coding Agent
// System prompt passed as parameter so compactor understands JS-ACT conversation format
// Uses ChatClientWithRetry for resilience as a background operation
function CodingAgentCompact(
  systemPrompt: string,
  messages: ChatMessage[],
  includeClaudeSpoof: bool,
  includeSystemPromptMessage: bool
) -> string {
  client ChatClientWithRetry
  prompt #"
    {%if includeClaudeSpoof%}
    {{ _.role("system") }}
    You are Claude Code, Anthropic's official CLI for Claude.
    {%endif%}
    {%if includeSystemPromptMessage%}
    {{ _.role("system") }}
    {{ systemPrompt }}
    {%endif%}

    {% for message in messages %}
      {{ _.role(message.role) }}
    {{ RenderContent(message.content) }}
    {% endfor %}

    {{ _.role("user", cache_control={"type": "ephemeral"}) }}
    COMPACTION REQUEST: You are NOT the coding agent right now. You are a COMPACTOR.
    Your job is to summarize the conversation above so the coding agent can continue working effectively.
    
    RESPOND WITH PLAIN TEXT ONLY - no JS-ACT code, no tool calls, no prose delimiters.
    
    Your summary MUST include:
    1. NARRATIVE: Brief history of what the user asked for and what was done
    2. CURRENT STATE: What has been built/changed, what files were modified
    3. ACTIVE GOALS: Current objectives, unresolved issues, next steps
    4. KEY DECISIONS: Important technical decisions and user preferences
    5. DATA AND VARIABLES: What data was fetched or created and into what variables (var names)
    
    Preserve INFORMATION over claims. Be specific about file paths, function names, and concrete details.
    Keep the summary concise but comprehensive - 3-5 paragraphs.
  "#
}