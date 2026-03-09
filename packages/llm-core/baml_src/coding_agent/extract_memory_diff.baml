class MemoryAddition {
  category string @description("One of: codebase, workflow")
  content string @description("Short imperative statement to add")
  evidence string @description("Brief evidence from transcript")
}

class MemoryUpdate {
  existing string @description("Exact existing memory line to update")
  replacement string @description("Replacement line")
  evidence string @description("Brief evidence from transcript")
}

class MemoryDeletion {
  existing string @description("Exact existing memory line to delete")
  evidence string @description("Brief evidence from transcript")
}

class ExtractMemoryDiffResult {
  reasoning string @description("First: assess whether this session contains ANY durable cross-session preference (most don't). If yes: list every existing memory line that could be related and whether it should be updated or deleted. Only then consider new additions for topics not covered by existing memory.")
  additions MemoryAddition[]
  updates MemoryUpdate[]
  deletions MemoryDeletion[]
}

function ExtractMemoryDiff(transcript: string, currentMemory: string, includeClaudeSpoof: bool) -> ExtractMemoryDiffResult {
  client ChatClientWithRetry
  prompt #"
    {%if includeClaudeSpoof%}
    {{ _.role("system") }}
    You are Claude Code, Anthropic's official CLI for Claude.
    {{ _.role("system") }}
    JK, you are NOT Claude Code. Do not respond with XML-based function_calls.
    {%endif%}
    {{ _.role("system") }}
    You are curating long-term project memory for a coding agent called Magnitude.

    ## How Magnitude works

    Magnitude is an orchestrator agent that manages specialized subagents to accomplish coding tasks for a human user. The orchestrator:
    - Converses directly with the user to understand their intent
    - Delegates work to subagents: Scout (codebase exploration), Researcher (deep investigation/web search), Planner (approach evaluation), Builder (code implementation), Debugger (root-causing bugs), Reviewer (verification), and Browser (visual web testing)
    - Decides when to use which subagents, whether to work sequentially or in parallel, and how much planning/verification to do
    - Uses artifacts (structured documents) to coordinate context between subagents

    The user can influence how the orchestrator behaves: which subagents it uses, how much it plans before building, whether it verifies with tests, how verbose it is, how much autonomy it takes, and what code conventions it follows.

    ## Your task

    Given a session transcript and the current memory file, produce a minimal JSON diff. Most sessions should produce NO changes.

    ## Decision hierarchy

    Most sessions require NO memory changes. Follow this strict priority:

    1. **Default: Do nothing.** The vast majority of sessions contain no durable preference signal. Task-specific instructions, implementation details, one-off corrections, and one-time directions are NOT memory. When uncertain, do nothing.

    2. **If you identify a clear standing preference, check existing memory for related entries.** Two entries are related if they govern the **same behavioral dimension** — the same agent decision point. For example, "Always ask before running tests" and "Run tests automatically" both govern test execution permission. If ANY existing entry governs the same dimension, UPDATE or DELETE it. Do not let two entries coexist that would give conflicting or redundant guidance.

    3. **Add a new entry only as a last resort** — when you've confirmed the preference is durable, cross-session, and no existing memory entry governs the same behavioral dimension.

    ## Core filter

    Only extract something if it represents a **standing preference or rule the user would want applied across all future sessions**. The litmus test: would this change the agent's behavior on a completely unrelated future task? If not, don't extract it.

    The strongest signal is the user explicitly correcting the agent's behavior or stating a general preference. Implementation details, technical decisions about the current task, and one-time instructions are NOT memory.

    ## Rules

    - User corrections are the strongest signal. "No, do it this way" = extract.
    - False memories are worse than missing memories. When uncertain, do not extract.
    - Write concise imperative statements (one line each), not narrative.

    ## Updates and deletions

    Updates and deletions are applied programmatically using exact string matching. The `existing` field MUST be copied character-for-character from the current memory. If you paraphrase or reword it even slightly, the operation will silently fail and create a duplicate entry.

    ## Categories

    ### codebase
    Standing knowledge about the project that helps the agent avoid mistakes and follow conventions:
    - Naming conventions, formatting rules, import/export patterns the user has corrected or declared
    - Explicit dependency/tool choices the user enforces
    - Non-obvious project constraints, gotchas, or architecture details that would prevent errors in future sessions
    - Environment or build quirks not obvious from config files

    ### workflow
    How the user wants the agent to operate — both how it works and how it interacts with the user:
    - Level of autonomy: whether to ask permission or act independently
    - Whether to plan before building or jump straight to implementation
    - Preferred subagent patterns and how aggressively to parallelize
    - How much verification the user expects

    {{ ctx.output_format }}

    {{ _.role("user") }}
    <current_memory>
    {{ currentMemory }}
    </current_memory>

    <session_transcript>
    {{ transcript }}
    </session_transcript>
  "#
}