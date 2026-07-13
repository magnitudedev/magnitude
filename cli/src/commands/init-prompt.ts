/**
 * Prompt content for the /init command.
 * Sent to the agent as a user message to trigger codebase exploration
 * and AGENTS.md generation.
 */

export const INIT_PROMPT = `Generate an AGENTS.md file for this project. AGENTS.md helps AI coding agents quickly understand and work effectively in a codebase without redundant exploration.

Thoroughly explore the codebase and produce a concise, actionable AGENTS.md (under 200 lines) in the project root. We want an agent reading this to understand: what the project is, what technologies it uses, how it's structured, how to build/test/run it, how the architecture works, and any important conventions or gotchas.

If AGENTS.md already exists, evaluate it and judge its comprehensiveness. If you can improve it, present your proposed changes and ask the user for confirmation before modifying the file.`
