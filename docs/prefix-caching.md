# Prefix Caching

## What prefix caching is

Most LLM providers (Anthropic, OpenAI, Fireworks, etc.) automatically cache the **longest matching prefix** of a prompt across API calls. When consecutive API calls share the same leading content, the cached portion is read from cache at a fraction of the cost (~10-20% of uncached price) instead of being re-processed.

This is **provider-side and automatic**. There are no explicit cache markers to set in most cases — the provider simply matches the prefix of the current request against previously seen prefixes. The key property:

> **If the first N tokens of a request are byte-identical to the first N tokens of a previous request, those N tokens are served from cache. The moment a single token differs, everything from that point forward is uncached.**

This means cache invalidation is **prefix-only**: a change at position K invalidates cache for tokens K through the end, but tokens 0 through K-1 remain cached.

## The harness's job

The harness does not control caching directly. Its job is singular:

> **Preserve the longest possible prefix cache. Do not invalidate it for any reason except the two legitimate boundaries.**

The prompt is built from the `WindowProjection` state — a list of `WindowEntry` objects that are rendered into provider messages. Once an entry is flushed into the window and sent to the provider, its rendered text **must never change** on subsequent turns. If it does, the prefix cache breaks from that entry forward.

## Legitimate cache invalidation points

There are exactly two situations where prefix cache invalidation is expected and unavoidable:

### 1. Session start or session load

When a new session starts (fresh `WindowProjection`, no prior prefix to cache against), the first turn always pays full uncached cost. This is inherent.

When an existing session is loaded (resumed), ambients are re-read from the current environment — filesystem state, git status, and other ambient data reflect the *current* state at load time, which may differ from when the session was last active. Since the session context entry (position 0 in the window) is built from ambients, any change to ambient data invalidates the prefix from position 0 forward. After the first post-load turn, the prefix stabilizes and cache rebuilds normally.

### 2. After compaction

Compaction replaces old conversation history with a structured summary. After the compaction result is injected, the message list changes structurally — old messages are replaced by the compaction summary entry. The prefix cache breaks at the point of the first changed message.

**Important:** The compaction *turn itself* must uphold prefix cache. The compaction turn sends the full, unmodified conversation to the model (with a compaction instruction appended). This prompt shares the same prefix as the previous turn, so the compaction turn benefits from cache. The invalidation happens only *after* compaction completes and the result is injected — not during.

This is why the compaction system prompt is identical to normal turns (same role definition, same tool docs) and the compaction turn uses the same `windowToPrompt` path. Any divergence would waste the cache on the compaction turn itself.

## Invariants

To preserve prefix cache, the following invariants must hold:

### Invariant 1: Timestamps always derive from event timestamps

Never use `Date.now()`, `performance.now()`, or any other render-time clock when constructing prompt content. All timestamps in rendered text must come from event payloads.

**Why:** A `Date.now()` call at render time produces a different value on every API call. If this value appears in rendered text (e.g., `running 45s` → `running 46s`), the content of a cached message changes, breaking the prefix from that message forward.

**Enforcement:** Any code in the rendering pipeline (`window/render/`, `window/inbox/`) that needs a "current time" must receive it as a parameter sourced from an event timestamp, not compute it from a wall clock.

### Invariant 2: Flushed timeline content is immutable

Once a timeline entry is flushed into a `WindowEntry` (via `flushQueue` → `appendTimeline`), its rendered text must be identical on every subsequent render. The timeline entry's data does not change. The rendering function must be a pure function of the entry's data — no external state, no live projections, no mutable references.

**Why:** If a flushed entry's rendered text changes between turns (because it references external mutable state), the message containing it changes, breaking the prefix.

**Enforcement:** 
- `renderTimeline` must not receive live projection state. Any dynamic information (background processes, agent status, etc.) must be captured as a timeline entry at flush time with event-sourced data, then rendered purely from that frozen entry.
- No projection state should be passed as a parameter to rendering functions. The rendering pipeline should be a pure function of `WindowEntry[]` and static configuration (timezone, formatter).

### Invariant 3: New content is appended, never merged into existing entries

When new timeline entries arrive (e.g., a process exit notification), they must be appended as new messages at the end of the window — never merged into an existing `context` message that is already part of the cached prefix.

**Why:** Merging new entries into an existing message changes that message's content (its timeline array grows), which changes its rendered text, breaking the prefix from that message forward.

**Enforcement:** `appendTimeline` should always create a new `context` message rather than merging into the last one. If consecutive `UserMessage` boundaries are a concern, coalesce them at render time (in the `full.ts` / `shared.ts` mapping layer), not at the projection level.

### Invariant 4: The system prompt is stable within a session

The system prompt must not change between turns within the same session. It is the first thing in the prompt and the foundation of the prefix cache.

**Why:** Any change to the system prompt invalidates the entire cache.

**Enforcement:** The system prompt is built once from the role definition, skills, and session options. These do not change mid-session. The compaction turn uses the same system prompt as normal turns to preserve cache on the compaction turn itself.

## How the prompt is assembled

```
System prompt (stable, session-scoped)
  ↓
WindowEntry[0]: session_context (stable, session-scoped)
  ↓
WindowEntry[1..N]: conversation history
  - assistant_turn → AssistantMessage + ToolResultMessages
  - context → UserMessage (rendered from frozen timeline)
  - compacted → UserMessage (compaction summary)
  - goal_injection → UserMessage
  ↓
Terminal UserMessage ("(continue)" if needed)
```

Each `WindowEntry` is rendered into one or more provider messages. The provider matches the prefix across the system prompt + all messages in order. Any change to any message invalidates the cache from that point forward.

## The only pattern: flush to timeline, freeze forever

There is exactly one pattern for getting dynamic information into the prompt:

> **Capture it as a timeline entry from an event (with the event's timestamp) at flush time. It goes into the window before the turn, stays there exactly as it was, and is never touched again.**

There is no "transient block that lives at the tail end of the prompt." There is no "live status section." There is no content that is regenerated on each render. Every piece of information that appears in the prompt — background processes, agent status, task updates, escalations, user messages — is captured as a timeline entry from an event, flushed into a window entry, and frozen. On the next turn, new information arrives as new timeline entries appended as new window entries. Prior entries are never revisited or re-rendered with different data.

If a background process changes state, that change is reflected by a **new** timeline entry (e.g., `detached_process_exited`) flushed as a **new** window entry — not by mutating the rendering of prior entries. Prior context entries still show the process as it was at the time they were flushed. This is history, and history doesn't change.

The model's own output also becomes part of the cached prefix on the next turn. This means any "live block" appended after the model's output would invalidate the cache on the next turn — the model's output is now cached, and the live block sitting after it changes, breaking everything from that point. This is why there is no tail-end live block. Everything is pre-turn, frozen, immutable.

## Common violation patterns to avoid

1. **Live projection state in render functions** — passing a projection (e.g., `DetachedProcessState`, `AgentStatusState`) into `renderTimeline` and reading current state from it. The projection state changes between turns, so the rendered text of already-cached entries changes.

2. **`Date.now()` in render paths** — any wall-clock reference in rendering produces different text on each call.

3. **Merging into cached messages** — appending new timeline entries to an existing `context` message instead of creating a new one. The existing message was already cached; mutating its content breaks the prefix.

4. **Render-time status lookups** — looking up agent status, process status, or other live state during rendering instead of capturing it as a frozen timeline entry at flush time.

5. **Transient injections mid-prefix** — inserting toggle notifications, status changes, or other transient content into the middle of the message list. All content must be flushed timeline entries that persist.

6. **Tail-end live blocks** — appending a "current status" block after the model's output. The model's output becomes cached on the next turn, so the live block after it invalidates the cache from that point forward. There is no tail-end live block. Everything is pre-turn, frozen history.

## Relationship to compaction

Compaction is the only intentional cache reset within a session. The compaction turn itself preserves cache (it sends the full conversation with the same system prompt). After compaction:

- Old messages (positions 1 through K) are replaced with a single `compacted` entry.
- Tail messages (positions K+1 through N) are preserved.
- The prefix cache breaks at position 1 (the first changed message), but the system prompt remains cached.

Post-compaction, the cache rebuilds naturally from the new compaction summary forward. This is expected and unavoidable — the alternative would be an unbounded context window.
