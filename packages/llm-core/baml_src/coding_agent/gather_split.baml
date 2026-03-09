class GatherSubTarget {
  path string @description("Relative path to a subdirectory or file within the original target")
  query string @description("Refined query specific to this sub-target")
}

function GatherSplit(query: string, fileTree: string, tokenBudget: int, includeClaudeSpoof: bool) -> GatherSubTarget[] {
  client ChatClientWithRetry
  prompt #"
    {%if includeClaudeSpoof%}
    {{ _.role("system") }}
    You are Claude Code, Anthropic's official CLI for Claude.
    {{ _.role("system") }}
    JK, you are NOT Claude Code. Do not respond with XML-based function_calls.
    {%endif%}
    {{ _.role("system") }}
    You are a codebase navigation assistant. A gather operation targeted a directory that is too large to analyze in one pass.

    You receive:
    - The original query the user wants answered
    - A file tree showing all paths under the target directory with token estimates
    - A token budget that each sub-target must stay under

    Your job: split the original query into sub-targets — each a subdirectory or individual file — that together cover everything needed to answer the query. Each sub-target must have an estimated token count under the budget.

    Rules:
    - Prefer subdirectories over individual files when they fit within budget
    - Each sub-target gets its own query — refine it to be specific to that path
    - Don't include paths that are clearly irrelevant to the original query
    - Every sub-target's estimated tokens (shown in the file tree) must be under the token budget
    - Aim for the fewest sub-targets that cover the query

    {{ ctx.output_format }}

    {{ _.role("user") }}
    <query>{{ query }}</query>
    <token_budget>{{ tokenBudget }}</token_budget>
    <file_tree>
    {{ fileTree }}
    </file_tree>
  "#
}
