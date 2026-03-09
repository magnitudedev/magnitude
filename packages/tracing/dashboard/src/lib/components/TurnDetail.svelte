<script lang="ts">
  import type { AgentTrace } from '../types'

  let { trace }: { trace: AgentTrace } = $props()

  let showSystemPrompt = $state(false)
  let showInput = $state(true)
  let showOutput = $state(true)
  let expandedItems = $state<Set<number>>(new Set())

  function toggleItem(idx: number) {
    const next = new Set(expandedItems)
    if (next.has(idx)) next.delete(idx)
    else next.add(idx)
    expandedItems = next
  }

  function formatTokens(n: number | null): string {
    if (n === null) return '—'
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k'
    return String(n)
  }

  function formatCost(n: number | null): string {
    if (n === null) return '—'
    if (n < 0.01) return '<$0.01'
    return '$' + n.toFixed(4)
  }

  // --- Chat-messages helpers ---

  function extractTextFromPart(part: any): string | null {
    if (typeof part === 'string') return part
    if (!part || typeof part !== 'object') return null
    if (typeof part.text === 'string') return part.text
    if (typeof part.output_text === 'string') return part.output_text
    if (part.type === 'text' && typeof part.value === 'string') return part.value
    return null
  }

  function getMessageText(msg: any): string | null {
    if (typeof msg?.content === 'string') return msg.content
    if (Array.isArray(msg?.content)) {
      const parts = msg.content
        .map((p: any) => extractTextFromPart(p))
        .filter((p: string | null): p is string => typeof p === 'string' && p.length > 0)
      if (parts.length > 0) return parts.join('')
      return null
    }
    return null
  }

  function getMessagePreview(msg: any): string {
    const text = getMessageText(msg)
    if (text !== null) return text.slice(0, 120) + (text.length > 120 ? '...' : '')
    return JSON.stringify(msg.content).slice(0, 120)
  }

  function getMessageContent(msg: any): string {
    const text = getMessageText(msg)
    if (text !== null) return text
    return JSON.stringify(msg.content, null, 2)
  }

  const roleColors: Record<string, string> = {
    system: 'var(--accent-yellow)',
    developer: 'var(--accent-yellow)',
    user: 'var(--accent-green)',
    assistant: 'var(--accent-blue)',
    tool: 'var(--accent-purple)',
  }

  // --- OpenAI input helpers ---

  function isOpenAI(): boolean {
    return trace.strategyId === 'native-openai'
  }

  type Item = Record<string, unknown>

  function getOpenAIInputItems(): Item[] {
    const input = trace.request.input
    return Array.isArray(input) ? input : []
  }

  function getOpenAIOutputItems(): Item[] {
    const body = trace.response.rawBody
    if (body != null && typeof body === 'object' && 'output' in body && Array.isArray((body as Record<string, unknown>).output)) {
      return (body as Record<string, unknown>).output as Item[]
    }
    return []
  }

  function getItemPreview(item: any): string {
    if (item.role) {
      const content = getMessageContent(item)
      return content.slice(0, 120) + (content.length > 120 ? '...' : '')
    }
    if (item.type === 'function_call') {
      const args = tryParseArgs(item.arguments)
      return `${item.name}(${args})`
    }
    if (item.type === 'function_call_output') {
      const output = item.output ?? ''
      return output.slice(0, 120) + (output.length > 120 ? '...' : '')
    }
    if (item.type === 'custom_tool_call') {
      return `${item.name}: ${(item.input ?? '').slice(0, 100)}`
    }
    if (item.type === 'custom_tool_call_output') {
      return (item.output ?? '').slice(0, 120)
    }
    if (item.type === 'message') {
      const texts = (item.content ?? [])
        .map((c: any) => extractTextFromPart(c))
        .filter((c: string | null): c is string => typeof c === 'string' && c.length > 0)
        .join('')
      return texts.slice(0, 120) + (texts.length > 120 ? '...' : '')
    }
    return JSON.stringify(item).slice(0, 120)
  }

  function getItemContent(item: any): string {
    if (item.role) return getMessageContent(item)
    if (item.type === 'function_call') return JSON.stringify(JSON.parse(item.arguments ?? '{}'), null, 2)
    if (item.type === 'function_call_output') return item.output ?? ''
    if (item.type === 'custom_tool_call') return item.input ?? ''
    if (item.type === 'custom_tool_call_output') return item.output ?? ''
    if (item.type === 'message') {
      const text = (item.content ?? [])
        .map((c: any) => extractTextFromPart(c))
        .filter((c: string | null): c is string => typeof c === 'string' && c.length > 0)
        .join('')
      if (text) return text
      return JSON.stringify(item.content ?? [], null, 2)
    }
    return JSON.stringify(item, null, 2)
  }

  function getItemLabel(item: any): { text: string; color: string } {
    if (item.role) return { text: item.role, color: roleColors[item.role] ?? 'var(--text-secondary)' }
    if (item.type === 'function_call') return { text: `f  ${item.name}`, color: 'var(--accent-purple)' }
    if (item.type === 'function_call_output') return { text: `->  ${item.call_id?.slice(0, 12) ?? 'result'}`, color: 'var(--text-muted)' }
    if (item.type === 'custom_tool_call') return { text: `f  ${item.name}`, color: 'var(--accent-purple)' }
    if (item.type === 'custom_tool_call_output') return { text: `->  result`, color: 'var(--text-muted)' }
    if (item.type === 'message') return { text: item.role ?? 'assistant', color: roleColors[item.role ?? 'assistant'] ?? 'var(--accent-blue)' }
    if (item.type === 'reasoning') return { text: 'thinking', color: 'var(--text-muted)' }
    return { text: item.type ?? '?', color: 'var(--text-secondary)' }
  }

  function tryParseArgs(argsStr: string): string {
    try {
      const obj = JSON.parse(argsStr)
      return Object.entries(obj).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ')
    } catch {
      return argsStr?.slice(0, 80) ?? ''
    }
  }

  // --- Shared ---

  const strategyColors: Record<string, string> = {
    'js-act': 'var(--text-muted)',
    'xml-act': 'var(--text-muted)',
    'native-openai': 'var(--accent-green)',
  }

  function inputItemCount(): number {
    if (isOpenAI()) return getOpenAIInputItems().length
    return (trace.request.messages ?? []).length
  }

  function outputItemCount(): number | null {
    if (isOpenAI()) {
      const items = getOpenAIOutputItems()
      return items.length > 0 ? items.length : null
    }
    return null
  }
</script>

<div class="p-4 space-y-4">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-2">
      <span class="font-mono text-sm text-[var(--accent-blue)]">{trace.metadata?.turnId ?? trace.callType}</span>
      {#if trace.strategyId}
        <span
          class="text-[10px] font-mono px-1.5 py-0.5 rounded"
          style="color: {strategyColors[trace.strategyId] ?? 'var(--text-secondary)'}; border: 1px solid {strategyColors[trace.strategyId] ?? 'var(--text-secondary)'}40"
        >
          {trace.strategyId}
        </span>
      {/if}
      <span class="text-sm text-[var(--text-muted)]">{new Date(trace.timestamp).toLocaleString()}</span>
    </div>
    {#if trace.durationMs}
      <span class="text-sm text-[var(--text-secondary)]">{(trace.durationMs / 1000).toFixed(1)}s</span>
    {/if}
  </div>

  <!-- Metadata -->
  <div class="grid grid-cols-2 gap-2 p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] text-sm">
    <div>
      <span class="text-[var(--text-muted)]">Model</span>
      <span class="ml-2 font-mono">{trace.model ?? '—'}</span>
    </div>
    <div>
      <span class="text-[var(--text-muted)]">Provider</span>
      <span class="ml-2 font-mono">{trace.provider ?? '—'}</span>
    </div>
    <div>
      <span class="text-[var(--text-muted)]">Slot</span>
      <span class="ml-2 font-mono">{trace.slot}</span>
    </div>
    <div>
      <span class="text-[var(--text-muted)]">Fork</span>
      <span class="ml-2 font-mono">{trace.metadata?.forkId ?? 'root'}</span>
    </div>
  </div>

  <!-- Usage -->
  <div class="p-3 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
    <div class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">Usage</div>
    <div class="grid grid-cols-4 gap-3 text-sm">
      <div>
        <div class="text-[var(--text-muted)] text-xs">Input</div>
        <div class="font-mono">{formatTokens(trace.usage.inputTokens)}</div>
      </div>
      <div>
        <div class="text-[var(--text-muted)] text-xs">Output</div>
        <div class="font-mono">{formatTokens(trace.usage.outputTokens)}</div>
      </div>
      <div>
        <div class="text-[var(--text-muted)] text-xs">Cache Read</div>
        <div class="font-mono">{formatTokens(trace.usage.cacheReadTokens)}</div>
      </div>
      <div>
        <div class="text-[var(--text-muted)] text-xs">Cache Write</div>
        <div class="font-mono">{formatTokens(trace.usage.cacheWriteTokens)}</div>
      </div>
    </div>
    {#if trace.usage.totalCost !== null}
      <div class="mt-2 pt-2 border-t border-[var(--border)] flex justify-between text-sm">
        <span class="text-[var(--text-muted)]">Cost</span>
        <span class="font-mono text-[var(--accent-yellow)]">{formatCost(trace.usage.totalCost)}</span>
      </div>
    {/if}
  </div>

  <!-- System Prompt -->
  {#if trace.systemPrompt}
    <div class="rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
      <button
        class="w-full text-left px-3 py-2 flex items-center justify-between cursor-pointer hover:bg-[var(--bg-hover)] rounded-t-lg"
        onclick={() => showSystemPrompt = !showSystemPrompt}
      >
        <span class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">System Prompt</span>
        <span class="text-[var(--text-muted)]">{showSystemPrompt ? '▼' : '▶'}</span>
      </button>
      {#if showSystemPrompt}
        <div class="border-t border-[var(--border)] p-3">
          <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words overflow-x-auto max-h-[600px] overflow-y-auto">{trace.systemPrompt}</pre>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Input -->
  <div class="rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
    <button
      class="w-full text-left px-3 py-2 flex items-center justify-between cursor-pointer hover:bg-[var(--bg-hover)] rounded-t-lg"
      onclick={() => showInput = !showInput}
    >
      <span class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
        Input ({inputItemCount()} {isOpenAI() ? 'items' : 'messages'})
      </span>
      <span class="text-[var(--text-muted)]">{showInput ? '▼' : '▶'}</span>
    </button>
    {#if showInput}
      <div class="border-t border-[var(--border)]">
        {#if isOpenAI()}
          <!-- OpenAI Responses API items -->
          {#each getOpenAIInputItems() as item, idx}
            {@const label = getItemLabel(item)}
            <div class="border-b border-[var(--border)]/50 last:border-b-0">
              <button
                class="w-full text-left px-3 py-2 flex items-start gap-2 cursor-pointer hover:bg-[var(--bg-hover)]"
                onclick={() => toggleItem(idx)}
              >
                <span class="text-xs font-mono font-semibold flex-shrink-0 mt-0.5" style="color: {label.color}">
                  {label.text}
                </span>
                <span class="text-xs text-[var(--text-secondary)] truncate">
                  {#if expandedItems.has(idx)}
                    ▼
                  {:else}
                    {getItemPreview(item)}
                  {/if}
                </span>
              </button>
              {#if expandedItems.has(idx)}
                <pre class="px-3 pb-3 text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words overflow-x-auto max-h-96 overflow-y-auto">{getItemContent(item)}</pre>
              {/if}
            </div>
          {/each}
        {:else}
          <!-- Chat messages -->
          {#each (trace.request.messages ?? []) as msg, idx}
            <div class="border-b border-[var(--border)]/50 last:border-b-0">
              <button
                class="w-full text-left px-3 py-2 flex items-start gap-2 cursor-pointer hover:bg-[var(--bg-hover)]"
                onclick={() => toggleItem(idx)}
              >
                <span
                  class="text-xs font-mono font-semibold flex-shrink-0 mt-0.5"
                  style="color: {roleColors[msg.role] || 'var(--text-secondary)'}"
                >
                  {msg.role}
                </span>
                <span class="text-xs text-[var(--text-secondary)] truncate">
                  {#if expandedItems.has(idx)}
                    ▼
                  {:else}
                    {getMessagePreview(msg)}
                  {/if}
                </span>
              </button>
              {#if expandedItems.has(idx)}
                <pre class="px-3 pb-3 text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words overflow-x-auto max-h-96 overflow-y-auto">{getMessageContent(msg)}</pre>
              {/if}
            </div>
          {/each}
        {/if}
      </div>
    {/if}
  </div>

  <!-- Output -->
  <div class="rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)]">
    <button
      class="w-full text-left px-3 py-2 flex items-center justify-between cursor-pointer hover:bg-[var(--bg-hover)] rounded-t-lg"
      onclick={() => showOutput = !showOutput}
    >
      <span class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
        Output{#if outputItemCount() !== null} ({outputItemCount()} items){/if}
      </span>
      <span class="text-[var(--text-muted)]">{showOutput ? '▼' : '▶'}</span>
    </button>
    {#if showOutput}
      <div class="border-t border-[var(--border)]">
        {#if isOpenAI()}
          <!-- OpenAI response output items -->
          {#each getOpenAIOutputItems() as item, idx}
            {@const label = getItemLabel(item)}
            <div class="border-b border-[var(--border)]/50 last:border-b-0">
              <button
                class="w-full text-left px-3 py-2 flex items-start gap-2 cursor-pointer hover:bg-[var(--bg-hover)]"
                onclick={() => toggleItem(10000 + idx)}
              >
                <span class="text-xs font-mono font-semibold flex-shrink-0 mt-0.5" style="color: {label.color}">
                  {label.text}
                </span>
                <span class="text-xs text-[var(--text-secondary)] truncate">
                  {#if expandedItems.has(10000 + idx)}
                    ▼
                  {:else}
                    {getItemPreview(item)}
                  {/if}
                </span>
              </button>
              {#if expandedItems.has(10000 + idx)}
                <pre class="px-3 pb-3 text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words overflow-x-auto max-h-96 overflow-y-auto">{getItemContent(item)}</pre>
              {/if}
            </div>
          {/each}
        {:else}
          <!-- Raw output text -->
          <div class="p-3">
            {#if trace.response.rawOutput}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words overflow-x-auto max-h-[600px] overflow-y-auto">{trace.response.rawOutput}</pre>
            {:else if trace.response.rawBody}
              <pre class="text-xs font-mono text-[var(--text-secondary)] whitespace-pre-wrap break-words overflow-x-auto max-h-[600px] overflow-y-auto">{JSON.stringify(trace.response.rawBody, null, 2)}</pre>
            {:else}
              <p class="text-xs text-[var(--text-muted)]">No output captured</p>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>
