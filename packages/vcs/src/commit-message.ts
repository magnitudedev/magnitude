/**
 * The commit message is an opaque string from VCS's perspective.
 * The agent layer gives it meaning (it's a tool call ID), but VCS
 * doesn't know or care — it just stores and retrieves it.
 */

/** Format a commit message. Identity function — the message is the string as-is. */
export function formatCommitMessage(message: string): string {
  return message
}

/** Parse a commit message. Returns the raw string. */
export function parseCommitMessage(message: string): { message?: string } {
  const trimmed = message.trim()
  if (!trimmed) return {}
  return { message: trimmed }
}
