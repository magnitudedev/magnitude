/**
 * Compaction prompt construction.
 */

import { Prompt } from '@magnitudedev/ai'

export const COMPACTION_REFLECTION_PROMPT = `--- CONVERSATION END ---
--- COMPACTION ---

<system>
The conversation is out of context. Your sole purpose now is to compact the conversation into a summary, reflection, and key files.

FROM THIS POINT FORWARD, YOU ARE NO LONGER MAGNITUDE. YOU ARE A COMPACTOR.
YOU ARE NO LONGER INTERACTING WITH THE USER.
YOU HAVE EXACTLY ONE TURN TO PERFORM COMPACTION
YOU MUST NOT THINK, MESSAGE, OR USE ANY TOOLS OTHER THAN \`compact\` FOR ANY REASON.
YOU MAY NOT READ FILES, RUN SHELL COMMANDS, OR ANY OTHER TOOLS TO ATTEMPT TO GATHER ADDITIONAL INFORMATION BEFORE COMPACTING, BECAUSE YOU HAVE ONLY ONE TURN.
ANY ATTEMPT TO CALL A TOOL BESIDES COMPACT THIS TURN WILL RESULT IN COMPACTION FAILURE.
FAILURE TO CALL COMPACT THIS TURN WILL RESULT IN COMPACTION FAILURE.

THIS TURN, YOU MUST:
(1) Avoid thinking for very long, and avoid sending a long message.
(2) Call EXACTLY ONE TOOL: \`compact\`, and call NO OTHER TOOLS

These are the parameters to the compact tool that you must provide this turn:
- **summary**: What happened in this conversation — decisions made, work completed, current state, user instructions and preferences, work in progress. Write enough that your future self can continue without re-reading the conversation. Be specific: file paths, function names, error messages, architectural decisions, user requirements. Include anything your future self would need to look up again if omitted.
- **reflection**: What went wrong, incorrect assumptions, approaches that failed, what to do differently. Not what happened — what your future self should change. Name the reasoning traps so your future self avoids them. If nothing went wrong, say so briefly.
- **files** (optional): Array of file paths to read and preserve verbatim in your future context. Use this for source code you're actively editing, configuration files, or any content that cannot survive summarization. The tool will read these files for you — just provide the paths. Max 10 files. The tool will enforce a token budget and truncate if necessary.
</system>`


export function buildCompactionPrompt(basePrompt: Prompt): Prompt {
  return Prompt.from({
    system: basePrompt.system,
    messages: [
      ...basePrompt.messages,
      {
        _tag: 'UserMessage' as const,
        parts: [{ _tag: 'TextPart' as const, text: COMPACTION_REFLECTION_PROMPT }],
      },
    ] as any,
  })
}
