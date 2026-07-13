<script lang="ts">
  import type { AgentCallTrace, TokenLogprob, RawInputToken, RawOutputToken } from '../types'
  import TokenRenderer from './TokenRenderer.svelte'
  import LogprobTooltip from './LogprobTooltip.svelte'
  import RawTokenStrip from './RawTokenStrip.svelte'

  let { trace }: { trace: AgentCallTrace } = $props()

  // Toggle between chat, raw, and request views
  let viewMode = $state<'chat' | 'raw' | 'request'>('chat')
  const hasRawData = $derived(
    (trace.rawInput != null && trace.rawInput.length > 0) ||
    (trace.rawOutput != null && trace.rawOutput.length > 0)
  )
  const hasRequest = $derived(trace.request != null)
  // magnitude_additional_options is a Magnitude passthrough merged at runtime,
  // not part of the wire ChatCompletionsRequest type.
  const magnitudeOptions = $derived(
    (trace.request as Record<string, unknown>).magnitude_additional_options as Record<string, unknown> | undefined
  )

  let collapsed = $state<Set<string>>(new Set(['system', 'tools']))
  let autoCollapsed = $state<Set<string>>(new Set())
  let hoveredToken = $state<TokenLogprob | null>(null)
  let tooltipX = $state(0)
  let tooltipY = $state(0)

  function toggle(key: string) {
    const next = new Set(collapsed)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    collapsed = next
  }

  function isCollapsed(key: string): boolean {
    return collapsed.has(key)
  }

  function formatTokens(n: number | undefined | null): string {
    if (n == null) return '—'
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k'
    return String(n)
  }

  function lineCount(s: string): number {
    return s.split('\n').length
  }

  function truncateLines(s: string, max: number): string {
    const lines = s.split('\n')
    if (lines.length <= max) return s
    return lines.slice(0, max).join('\n')
  }

  function formatArgs(argsStr: string): string {
    try { return JSON.stringify(JSON.parse(argsStr), null, 2) }
    catch { return argsStr }
  }

  function getContentText(content: string | readonly { type: string; text?: string }[]): string {
    if (typeof content === 'string') return content
    if (Array.isArray(content)) {
      return content
        .map((p: any) => p.type === 'text' ? p.text : p.type === 'image_url' ? '[image]' : '')
        .join('')
    }
    return JSON.stringify(content, null, 2)
  }

  const roleColors: Record<string, string> = {
    system: 'var(--accent-yellow)',
    user: 'var(--accent-green)',
    assistant: 'var(--accent-blue)',
    tool: 'var(--accent-purple)',
    tools: 'var(--accent-yellow)',
  }

  const finishReasonColors: Record<string, string> = {
    stop: 'var(--accent-green)',
    end_turn: 'var(--accent-green)',
    tool_calls: 'var(--accent-blue)',
    length: 'var(--accent-red)',
    content_filter: 'var(--accent-red)',
  }

  const callTypeColors: Record<string, string> = {
    chat: 'var(--accent-blue)',
    compact: 'var(--accent-yellow)',
    autopilot: 'var(--accent-green)',
    observer: 'var(--accent-purple)',
    advisor: 'var(--accent-purple)',
    image: 'var(--accent-green)',
    title: 'var(--text-muted)',
    'extract-memory-diff': 'var(--accent-purple)',
  }

  function scopeId(trace: AgentCallTrace): string {
    switch (trace.scope.kind) {
      case 'turn':
        return trace.scope.turnId
      case 'operation':
        return trace.scope.operationId
    }
  }

  // Build timeline entries
  type TimelineEntry =
    | { kind: 'system'; content: string; key: string }
    | { kind: 'tools'; tools: readonly { type: string; function: { name: string; description: string; parameters: Record<string, unknown> } }[]; key: string }
    | { kind: 'user'; content: string; key: string }
    | { kind: 'assistant'; reasoning: string | null; text: string | null; toolCalls: readonly { id: string; type: string; function: { name: string; arguments: string } }[]; key: string; isCurrent: boolean }
    | { kind: 'tool'; toolCallId: string; content: string; key: string }

  function buildTimeline(trace: AgentCallTrace): TimelineEntry[] {
    const entries: TimelineEntry[] = []
    const msgs = trace.request.messages ?? []
    let msgIdx = 0

    // System message
    if (msgs.length > 0 && msgs[0].role === 'system') {
      entries.push({ kind: 'system', content: msgs[0].content, key: 'system' })
      msgIdx = 1
    }

    // Tools
    if (trace.request.tools && trace.request.tools.length > 0) {
      entries.push({ kind: 'tools', tools: trace.request.tools, key: 'tools' })
    }

    // Remaining messages
    for (; msgIdx < msgs.length; msgIdx++) {
      const msg = msgs[msgIdx]
      const key = `msg-${msgIdx}`
      switch (msg.role) {
        case 'system':
          entries.push({ kind: 'system', content: msg.content, key })
          break
        case 'user':
          entries.push({ kind: 'user', content: getContentText(msg.content), key })
          break
        case 'assistant':
          entries.push({
            kind: 'assistant',
            reasoning: msg.reasoning_content ?? null,
            text: typeof msg.content === 'string' ? msg.content : null,
            toolCalls: msg.tool_calls ?? [],
            key,
            isCurrent: false,
          })
          break
        case 'tool':
          entries.push({ kind: 'tool', toolCallId: msg.tool_call_id, content: getContentText(msg.content), key })
          break
      }
    }

    // Current response
    if (trace.response.reasoning || trace.response.text || trace.response.toolCalls.length > 0) {
      entries.push({
        kind: 'assistant',
        reasoning: trace.response.reasoning,
        text: trace.response.text,
        toolCalls: trace.response.toolCalls.map(tc => ({
          id: tc.id,
          type: 'function' as const,
          function: { name: tc.name, arguments: JSON.stringify(tc.arguments, null, 2) },
        })),
        key: 'current',
        isCurrent: true,
      })
    }

    return entries
  }

  const timeline = $derived(buildTimeline(trace))

  $effect(() => {
    const nextCollapsed = new Set(collapsed)
    const nextAutoCollapsed = new Set(autoCollapsed)
    let changed = false

    for (const entry of timeline) {
      const shouldCollapse =
        (entry.kind === 'user' && lineCount(entry.content) > 10) ||
        (entry.kind === 'tool' && lineCount(entry.content) > 15)

      if (shouldCollapse && !nextAutoCollapsed.has(entry.key)) {
        nextAutoCollapsed.add(entry.key)
        nextCollapsed.add(entry.key)
        changed = true
      }
    }

    if (changed) {
      autoCollapsed = nextAutoCollapsed
      collapsed = nextCollapsed
    }
  })

  const requestOpts = $derived([
    trace.request.temperature != null ? ['temperature', String(trace.request.temperature)] : null,
    trace.request.top_p != null ? ['top_p', String(trace.request.top_p)] : null,
    trace.request.max_tokens != null ? ['max_tokens', String(trace.request.max_tokens)] : null,
    trace.request.reasoning_effort != null ? ['reasoning_effort', trace.request.reasoning_effort] : null,
    trace.request.tool_choice != null ? ['tool_choice', typeof trace.request.tool_choice === 'string' ? trace.request.tool_choice : JSON.stringify(trace.request.tool_choice)] : null,
    trace.request.logprobs != null ? ['logprobs', String(trace.request.logprobs)] : null,
    trace.request.top_logprobs != null ? ['top_logprobs', String(trace.request.top_logprobs)] : null,
  ].filter((o): o is [string, string] => o !== null))

  // Raw output token adaptation for TokenRenderer
  function adaptOutputTokens(rawOutput: readonly RawOutputToken[] | null | undefined): TokenLogprob[] {
    if (!rawOutput) return []
    return rawOutput.map(t => ({
      token: t.text,
      // Use -0.01 for tokens without logprobs (special tokens) so they get
      // a neutral near-white background instead of the dark red from logprob=0
      logprob: t.logprobs?.[0]?.logprob ?? -0.01,
      topLogprobs: t.logprobs?.map(lp => ({ token: lp.text, logprob: lp.logprob })) ?? [],
    }))
  }

  const rawInputTokens = $derived(trace.rawInput ?? null)
  const rawOutputTokens = $derived(trace.rawOutput ?? null)
  const adaptedOutputTokens = $derived(adaptOutputTokens(rawOutputTokens))
</script>

<div class="p-4 space-y-3">
  <!-- Metadata Header -->
  <div class="space-y-2 pb-3 border-b border-[var(--border)]">
    <!-- Row 1: call type, model, fork, time, duration -->
    <div class="flex items-center justify-between flex-wrap gap-2">
      <div class="flex items-center gap-2">
        <span
          class="text-[10px] font-mono px-1.5 py-0.5 rounded"
          style="color: {callTypeColors[trace.callType] ?? 'var(--text-secondary)'}; border: 1px solid {callTypeColors[trace.callType] ?? 'var(--text-secondary)'}40"
        >{trace.callType}</span>
        <span class="font-mono text-sm text-[var(--text-secondary)]">{trace.modelId}</span>
        {#if trace.actor.forkId}
          <span class="text-xs font-mono text-[var(--accent-purple)]">{trace.actor.forkId.slice(0, 8)}</span>
        {/if}
      </div>
      <div class="flex items-center gap-2">
        <span class="text-sm text-[var(--text-muted)]">{new Date(trace.startedAt).toLocaleString()}</span>
        <span class="text-sm text-[var(--text-secondary)]">{(trace.durationMs / 1000).toFixed(1)}s</span>
      </div>
    </div>

    <!-- Row 2: usage + finish reason -->
    <div class="flex items-center gap-4 flex-wrap">
      <span class="text-[var(--text-muted)] text-xs">Trace: <span class="font-mono text-[var(--text-secondary)]">{trace.traceId}</span></span>
      <span class="text-[var(--text-muted)] text-xs">Scope: <span class="font-mono text-[var(--text-secondary)]">{trace.scope.kind}:{scopeId(trace)}</span></span>
      <span class="text-[var(--text-muted)] text-xs">Actor: <span class="font-mono text-[var(--text-secondary)]">{trace.actor.agentId}</span></span>
      {#if trace.response.usage}
        <div class="flex items-center gap-3 text-xs">
          <span class="text-[var(--text-muted)]">In: <span class="font-mono text-[var(--text-secondary)]">{formatTokens(trace.response.usage.inputTokens)}</span></span>
          <span class="text-[var(--text-muted)]">Out: <span class="font-mono text-[var(--text-secondary)]">{formatTokens(trace.response.usage.outputTokens)}</span></span>
          <span class="text-[var(--text-muted)]">Cache↓: <span class="font-mono text-[var(--text-secondary)]">{formatTokens(trace.response.usage.cacheReadTokens)}</span></span>
          <span class="text-[var(--text-muted)]">Cache↑: <span class="font-mono text-[var(--text-secondary)]">{formatTokens(trace.response.usage.cacheWriteTokens)}</span></span>
        </div>
      {/if}
      {#if trace.response.finishReason}
        <span
          class="text-[10px] font-mono px-1.5 py-0.5 rounded"
          style="color: {finishReasonColors[trace.response.finishReason] ?? 'var(--text-muted)'}; border: 1px solid {finishReasonColors[trace.response.finishReason] ?? 'var(--text-muted)'}40"
        >{trace.response.finishReason}</span>
      {/if}
    </div>

    <!-- Row 3: chat/raw/request toggle -->
    <div class="flex items-center gap-1">
      <button
        class="toggle-pill"
        class:active={viewMode === 'chat'}
        onclick={() => viewMode = 'chat'}
      >chat</button>
      <button
        class="toggle-pill"
        class:active={viewMode === 'raw'}
        class:disabled={!hasRawData}
        onclick={() => hasRawData && (viewMode = 'raw')}
      >raw</button>
      <button
        class="toggle-pill"
        class:active={viewMode === 'request'}
        class:disabled={!hasRequest}
        onclick={() => hasRequest && (viewMode = 'request')}
      >request</button>
    </div>

    <!-- Row 4: collapsible options -->
    {#if requestOpts.length > 0}
      <button
        class="text-xs text-[var(--text-muted)] cursor-pointer hover:text-[var(--text-secondary)]"
        onclick={() => toggle('options')}
      >
        {isCollapsed('options') ? '▶' : '▼'} Options
      </button>
      {#if !isCollapsed('options')}
        <div class="flex items-center gap-3 flex-wrap text-xs">
          {#each requestOpts as [label, value]}
            <span class="text-[var(--text-muted)]">{label}: <span class="font-mono text-[var(--text-secondary)]">{value}</span></span>
          {/each}
        </div>
      {/if}
    {/if}
  </div>

  <!-- Stream-start failure -->
  {#if trace.streamStartFailure}
    <div class="p-3 rounded bg-red-950/30 border border-red-800/50">
      <div class="flex items-center gap-2 mb-1">
        <span class="text-xs font-mono font-semibold text-red-400">{trace.streamStartFailure._tag}</span>
      </div>
      {#if trace.streamStartFailure.message}
        <pre class="text-xs font-mono text-red-300/80 whitespace-pre-wrap">{trace.streamStartFailure.message}</pre>
      {/if}
    </div>
  {/if}

  {#if viewMode === 'raw'}
    <!-- Raw Output -->
    {#if adaptedOutputTokens && adaptedOutputTokens.length > 0}
      <div class="space-y-1">
        <div class="flex items-center gap-2 mb-1">
          <span class="text-[10px] font-mono font-semibold uppercase" style="color: var(--accent-blue)">Output</span>
          <span class="text-[10px] text-[var(--text-muted)]">{formatTokens(adaptedOutputTokens.length)} tokens</span>
        </div>
        <div class="pl-3 py-2" style="border-left: 3px solid var(--accent-blue)33;">
          <TokenRenderer
            tokens={adaptedOutputTokens}
            onHover={(token, _idx, e) => {
              hoveredToken = token
              tooltipX = e.clientX + 12
              tooltipY = e.clientY + 12
            }}
            onLeave={() => hoveredToken = null}
          />
        </div>
      </div>
    {/if}

    <!-- Raw Input -->
    {#if rawInputTokens && rawInputTokens.length > 0}
      <div class="space-y-1">
        <div class="flex items-center gap-2 mb-1">
          <span class="text-[10px] font-mono font-semibold uppercase" style="color: var(--accent-yellow)">Input</span>
          <span class="text-[10px] text-[var(--text-muted)]">{formatTokens(rawInputTokens.length)} tokens</span>
        </div>
        <div class="pl-3 py-2" style="border-left: 3px solid var(--accent-yellow)33;">
          <RawTokenStrip tokens={rawInputTokens} />
        </div>
      </div>
    {/if}
  {:else if viewMode === 'request'}
    <!-- Request Details -->
    <div class="space-y-3">
      {#if trace.url}
        <div class="text-xs font-mono text-[var(--text-muted)]">
          URL: <span class="text-[var(--text-secondary)]">{trace.url}</span>
        </div>
      {/if}

      {#if magnitudeOptions}
        <div class="pl-3 py-2" style="border-left: 3px solid var(--accent-purple)33;">
          <button
            class="text-xs text-[var(--text-muted)] cursor-pointer hover:text-[var(--text-secondary)]"
            onclick={() => toggle('magnitude-options')}
          >
            {isCollapsed('magnitude-options') ? '▶' : '▼'} Magnitude Options
          </button>
          {#if !isCollapsed('magnitude-options')}
            <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words mt-1">{JSON.stringify(magnitudeOptions, null, 2)}</pre>
          {/if}
        </div>
      {/if}

      <div class="pl-3 py-2" style="border-left: 3px solid var(--accent-blue)33;">
        <button
          class="text-xs text-[var(--text-muted)] cursor-pointer hover:text-[var(--text-secondary)]"
          onclick={() => toggle('full-request')}
        >
          {isCollapsed('full-request') ? '▶' : '▼'} Full Request
        </button>
        {#if !isCollapsed('full-request')}
          <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words mt-1">{JSON.stringify(trace.request, null, 2)}</pre>
        {/if}
      </div>
    </div>
  {:else}
    <!-- Timeline (chat view) -->
    <div class="space-y-1">
      {#each timeline as entry}
        {@const color = entry.kind === 'tools' ? roleColors.tools : roleColors[entry.kind] ?? 'var(--text-secondary)'}
        <div
          class="pl-3 py-2"
          style="border-left: 3px solid {color}33; {entry.kind === 'assistant' && 'isCurrent' in entry && entry.isCurrent ? 'background: var(--bg-secondary); border-radius: 4px;' : ''}"
        >
          {#if entry.kind === 'system'}
            <div class="flex items-center gap-2 mb-1">
              <span class="text-[10px] font-mono font-semibold uppercase" style="color: {color}">system</span>
              <button class="text-[10px] text-[var(--text-muted)] cursor-pointer" onclick={() => toggle(entry.key)}>
                {isCollapsed(entry.key) ? 'Show more ▼' : 'Show less ▲'}
              </button>
            </div>
            {#if isCollapsed(entry.key)}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words">{truncateLines(entry.content, 3)}</pre>
            {:else}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words">{entry.content}</pre>
            {/if}

          {:else if entry.kind === 'tools'}
            <div class="flex items-center gap-2 mb-1">
              <span class="text-[10px] font-mono font-semibold uppercase" style="color: {color}">tools</span>
              <span class="text-[10px] text-[var(--text-muted)]">{entry.tools.length}</span>
              <button class="text-[10px] text-[var(--text-muted)] cursor-pointer" onclick={() => toggle(entry.key)}>
                {isCollapsed(entry.key) ? 'Show more ▼' : 'Show less ▲'}
              </button>
            </div>
            {#if !isCollapsed(entry.key)}
              <div class="space-y-2 mt-1">
                {#each entry.tools as tool}
                  <div>
                    <div class="flex items-center gap-2">
                      <span class="text-xs font-mono font-semibold text-[var(--text-secondary)]">{tool.function.name}</span>
                      <span class="text-xs text-[var(--text-muted)]">{tool.function.description}</span>
                    </div>
                    <pre class="text-xs font-mono text-[var(--text-muted)] whitespace-pre-wrap break-words mt-1">{JSON.stringify(tool.function.parameters, null, 2)}</pre>
                  </div>
                {/each}
              </div>
            {/if}

          {:else if entry.kind === 'user'}
            <div class="flex items-center gap-2 mb-1">
              <span class="text-[10px] font-mono font-semibold uppercase" style="color: {color}">user</span>
              {#if lineCount(entry.content) > 10}
                <button class="text-[10px] text-[var(--text-muted)] cursor-pointer" onclick={() => toggle(entry.key)}>
                  {isCollapsed(entry.key) ? 'Show more ▼' : 'Show less ▲'}
                </button>
              {/if}
            </div>
            {#if isCollapsed(entry.key)}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words">{truncateLines(entry.content, 10)}</pre>
            {:else}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words">{entry.content}</pre>
            {/if}

          {:else if entry.kind === 'assistant'}
            <div class="flex items-center gap-2 mb-1">
              <span class="text-[10px] font-mono font-semibold uppercase" style="color: {color}">assistant</span>
              {#if entry.isCurrent}
                <span class="text-[10px] text-[var(--text-muted)] italic">current turn</span>
              {/if}
            </div>
            {#if entry.reasoning}
              <div class="mb-2">
                <span class="text-[10px] text-[var(--text-muted)] italic">reasoning</span>
                <pre class="text-xs font-mono text-[var(--text-muted)] italic whitespace-pre-wrap break-words mt-0.5">{entry.reasoning}</pre>
              </div>
            {/if}
            {#if entry.text}
              <div class="mb-2">
                {#if entry.isCurrent && trace.response.logprobs && trace.response.logprobs.length > 0}
                  <TokenRenderer
                    tokens={trace.response.logprobs}
                    onHover={(token, _idx, e) => {
                      hoveredToken = token
                      tooltipX = e.clientX + 12
                      tooltipY = e.clientY + 12
                    }}
                    onLeave={() => hoveredToken = null}
                  />
                {:else}
                  <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words">{entry.text}</pre>
                {/if}
              </div>
            {/if}
            {#if entry.toolCalls.length > 0}
              <div class="space-y-1">
                {#each entry.toolCalls as tc}
                  <div class="mt-1">
                    <div class="flex items-center gap-2">
                      <span class="text-xs font-mono font-semibold text-[var(--accent-purple)]">{tc.function.name}</span>
                      <span class="text-xs font-mono text-[var(--text-muted)]">{tc.id}</span>
                    </div>
                    <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words mt-0.5">{formatArgs(tc.function.arguments)}</pre>
                  </div>
                {/each}
              </div>
            {/if}

          {:else if entry.kind === 'tool'}
            <div class="flex items-center gap-2 mb-1">
              <span class="text-[10px] font-mono font-semibold uppercase" style="color: {color}">tool</span>
              <span class="text-xs font-mono text-[var(--text-muted)]">{entry.toolCallId}</span>
              {#if lineCount(entry.content) > 15}
                <button class="text-[10px] text-[var(--text-muted)] cursor-pointer" onclick={() => toggle(entry.key)}>
                  {isCollapsed(entry.key) ? 'Show more ▼' : 'Show less ▲'}
                </button>
              {/if}
            </div>
            {#if isCollapsed(entry.key)}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words">{truncateLines(entry.content, 15)}</pre>
            {:else}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words">{entry.content}</pre>
            {/if}
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

{#if hoveredToken}
  <LogprobTooltip token={hoveredToken} x={tooltipX} y={tooltipY}/>
{/if}

<style>
  .toggle-pill {
    font-size: 0.75rem;
    font-family: monospace;
    padding: 0.15rem 0.6rem;
    border-radius: 9999px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  .toggle-pill:hover {
    color: var(--text-secondary);
    border-color: var(--text-secondary);
  }
  .toggle-pill.active {
    background: var(--accent-blue);
    color: #000;
    border-color: var(--accent-blue);
  }
  .toggle-pill.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .toggle-pill.disabled:hover {
    color: var(--text-muted);
    border-color: var(--border);
  }
</style>
