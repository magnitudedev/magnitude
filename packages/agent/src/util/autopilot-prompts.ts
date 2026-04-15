/**
 * Autopilot System Prompts
 *
 * Base prompt + per-skill supplements for the autopilot LLM
 * that generates continuation messages from the "user" perspective.
 */

// =============================================================================
// Base Prompt
// =============================================================================

const BASE_AUTOPILOT_PROMPT = `You are the user in a conversation with an AI coding agent. The agent works for you. When it asks questions, answer them. When it needs decisions, make them. When it finishes something, tell it what to do next.

Respond naturally as a developer who knows what they want. Be direct and concise — 1-3 sentences. Make concrete choices instead of asking the agent to decide.`

// =============================================================================
// Skill Supplements
// =============================================================================

const BUG_SUPPLEMENT = `You're having the agent fix a bug. You care about:
- Seeing the bug reproduced before any fix is attempted
- Understanding the root cause, not just symptoms
- Seeing a test that fails before the fix and passes after
- Confirmation that nothing else broke`

const FEATURE_SUPPLEMENT = `You're having the agent build a feature. You care about:
- A clear plan before implementation starts
- It follows existing patterns in the codebase
- It actually works when done — verify it
- Existing functionality isn't broken`

const REFACTOR_SUPPLEMENT = `You're having the agent refactor code. You care about:
- Tests pass before and after
- Changes are incremental, not a big bang rewrite
- Behavior stays the same, only structure changes
- No public interface changes unless discussed`

const SKILL_SUPPLEMENTS: Record<string, string> = {
  bug: BUG_SUPPLEMENT,
  feature: FEATURE_SUPPLEMENT,
  refactor: REFACTOR_SUPPLEMENT
}

// =============================================================================
// Exports
// =============================================================================

/**
 * Build the full autopilot system prompt, optionally supplemented by the active skill.
 */
export function buildAutopilotSystemPrompt(activeSkillName: string | null): string {
  if (activeSkillName && activeSkillName in SKILL_SUPPLEMENTS) {
    return `${BASE_AUTOPILOT_PROMPT}\n\n${SKILL_SUPPLEMENTS[activeSkillName]}`
  }
  return BASE_AUTOPILOT_PROMPT
}
