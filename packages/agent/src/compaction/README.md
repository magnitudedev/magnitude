# Compaction

Compaction replaces old conversation history with a structured summary when the context window gets too large, allowing the agent to continue working without losing critical information.

## Trigger

Compaction triggers when the estimated token count crosses the **soft cap** (default: 90% of hard cap). The hard cap is `model.contextWindow - 8192` (output token reserve).

## What happens during compaction

1. **Message selection**: The system determines which messages to compact. Recent messages are kept (the tail 10% of soft cap worth of tokens). Everything else (excluding the session context at position 0) is compacted.

2. **Compaction turn**: An agent turn runs with the full conversation visible. The agent is instructed to call the `compact()` tool with:
   - **summary**: What happened — decisions, work completed, current state, user preferences, work in progress. Specific enough for the future self to continue without re-reading.
   - **reflection**: What went wrong, incorrect assumptions, what to do differently.
   - **files** (optional): Up to 10 file paths to preserve. Files ≤10k chars are included verbatim. Files >10k chars are listed as references with their character count.

3. **Retry on failure to call compact()**: If the agent doesn't call `compact()`, the turn is retried up to 3 times.

4. **Fallback**: If all 3 retries fail (agent never calls `compact()`), the system falls back to **raw tail preservation** — keeps the most recent 25% of soft cap worth of messages, discards the rest. No summary is generated.

## Injection

After the compaction turn completes, the result is injected into the conversation window:

- **Structured compaction** (agent called `compact()`): The old messages are replaced with a single `<compaction_summary>` block containing the summary, reflection, and key files. Recent tail messages are preserved after it.

- **Fallback** (agent failed to call `compact()`): The old messages are simply dropped. The most recent messages that fit within 25% of soft cap are kept.

In both cases, the session context (message 0) is always preserved.

## Timing

Compaction injection waits for the main agent turn to be idle. If a turn is active when compaction finishes, injection is deferred until the turn completes. This prevents the window from being rewritten under an active turn.

## Post-injection

After injection, the system recalculates the token estimate. If the conversation is still above the soft cap, compaction triggers again immediately. This repeats until the context fits.

## Failure handling

If the compaction turn itself fails (model error, timeout, connection error):
- Compaction retries immediately. The conversation is still too long — the problem hasn't gone away.
- If the main turn was blocked at the hard cap, it stays blocked. Unblocking would let the main turn proceed into guaranteed failure.
- Connection errors use exponential backoff (same as normal turns).

## System prompt caching

The compaction turn uses the same system prompt as normal turns (same role definition, same tool docs). This preserves provider-level prompt cache hits. The `compact()` tool is included in all role toolkits so the system prompt is identical between normal and compaction turns.

## Token budget

After compaction, the window contains: system prompt + session context + compaction summary + kept tail messages. All of this must fit within the soft cap. The `compact()` tool limits the total size of the agent's output (summary + reflection + file contents) to the remaining space after the other pieces are accounted for.

## Tracing

Compaction model calls are traced with `callType: "compact"` to distinguish them from normal chat turns in the tracing dashboard.
