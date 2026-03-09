<script lang="ts">
  import { traceStore } from '../stores/traces.svelte'
  import TurnDetail from './TurnDetail.svelte'

  let {
    sessionId,
    selectedTurnIdFromRoute = null,
    onSelectTurn,
    onBack,
  }: {
    sessionId: string
    selectedTurnIdFromRoute?: string | null
    onSelectTurn?: (turnId: string | null, replace?: boolean) => void
    onBack: () => void
  } = $props()

  $effect(() => {
    if (sessionId) {
      traceStore.selectSession(sessionId)
    }
  })

  $effect(() => {
    if (!sessionId || traceStore.selectedSessionId !== sessionId || traceStore.loading) return

    if (!selectedTurnIdFromRoute) {
      if (traceStore.selectedTurnId !== null) traceStore.selectTurn(null)
      return
    }

    const exists = traceStore.allTracesSorted.some(
      (trace) => (trace.metadata?.turnId || trace.timestamp) === selectedTurnIdFromRoute,
    )

    if (exists) {
      if (traceStore.selectedTurnId !== selectedTurnIdFromRoute) {
        traceStore.selectTurn(selectedTurnIdFromRoute)
      }
    } else if (traceStore.allTracesSorted.length > 0) {
      traceStore.selectTurn(null)
      onSelectTurn?.(null, true)
    }
  })

  function formatTokens(n: number): string {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M'
    if (n >= 1_000) return (n / 1_000).toFixed(1) + 'k'
    return String(n)
  }

  function formatCost(n: number): string {
    if (n < 0.01) return '<$0.01'
    return '$' + n.toFixed(2)
  }

  function formatTime(ts: string): string {
    try {
      return new Date(ts).toLocaleTimeString()
    } catch {
      return ts
    }
  }

  const callTypeColors: Record<string, string> = {
    chat: 'var(--accent-blue)',
    compact: 'var(--accent-yellow)',
    autopilot: 'var(--accent-green)',
    title: 'var(--text-muted)',
  }

  function getCallTypeColor(type: string | undefined): string {
    return callTypeColors[type ?? 'chat'] ?? 'var(--text-secondary)'
  }
</script>

<div class="flex flex-col h-screen">
  <!-- Header -->
  <header class="border-b border-[var(--border)] px-4 py-2 flex items-center gap-4 flex-shrink-0">
    <button class="text-sm text-[var(--accent-blue)] hover:underline cursor-pointer" onclick={onBack}>← Sessions</button>
    <span class="text-sm font-mono text-[var(--text-secondary)]">{sessionId}</span>
  </header>

  <div class="flex flex-1 overflow-hidden">
  <!-- Left sidebar: filters -->
  <div class="w-64 border-r border-[var(--border)] flex flex-col overflow-y-auto">
    <!-- Stats -->
    <div class="p-4 border-b border-[var(--border)]">
      <div class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)] mb-2">Overview</div>
      <div class="space-y-1 text-sm">
        <div class="flex justify-between">
          <span class="text-[var(--text-secondary)]">Traces</span>
          <span class="font-mono">{traceStore.traces.length}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-[var(--text-secondary)]">Input tokens</span>
          <span class="font-mono">{formatTokens(traceStore.totalTokens.input)}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-[var(--text-secondary)]">Output tokens</span>
          <span class="font-mono">{formatTokens(traceStore.totalTokens.output)}</span>
        </div>
        {#if traceStore.totalCost > 0}
          <div class="flex justify-between">
            <span class="text-[var(--text-secondary)]">Cost</span>
            <span class="font-mono text-[var(--accent-yellow)]">{formatCost(traceStore.totalCost)}</span>
          </div>
        {/if}
      </div>
    </div>

    <!-- Call Type filter -->
    <div class="p-4 border-b border-[var(--border)]">
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">Call Type</div>
        <button class="text-xs text-[var(--accent-blue)] hover:underline cursor-pointer" onclick={() => traceStore.showAllCallTypes()}>All</button>
      </div>
      <div class="space-y-1">
        {#each traceStore.callTypes as ct}
          <button
            class="w-full text-left px-2 py-1 rounded text-xs flex items-center gap-2 cursor-pointer transition-colors {traceStore.hiddenCallTypes.has(ct) ? 'text-[var(--text-muted)]' : 'text-[var(--text-primary)]'}"
            onclick={() => traceStore.toggleCallType(ct)}
          >
            <span class="inline-block w-2 h-2 rounded-full flex-shrink-0" style="background: {getCallTypeColor(ct)}; opacity: {traceStore.hiddenCallTypes.has(ct) ? 0.3 : 1}"></span>
            {ct}
          </button>
        {/each}
      </div>
    </div>

    <!-- Forks filter -->
    <div class="p-4">
      <div class="flex items-center justify-between mb-2">
        <div class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">Forks</div>
        <button class="text-xs text-[var(--accent-blue)] hover:underline cursor-pointer" onclick={() => traceStore.showAllForks()}>All</button>
      </div>
      <div class="space-y-1">
        {#each [...traceStore.availableForks] as [forkId, info]}
          <button
            class="w-full text-left px-2 py-1 rounded text-xs flex items-center justify-between cursor-pointer transition-colors {traceStore.hiddenForkIds.has(forkId) ? 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)]' : 'text-[var(--text-primary)] bg-[var(--bg-hover)]'}"
            onclick={() => traceStore.toggleFork(forkId)}
          >
            <span class="font-mono" title={forkId ?? 'root'}>{info.name}</span>
            <span class="text-[var(--text-muted)]">{info.count}</span>
          </button>
        {/each}
      </div>
    </div>
  </div>

  <!-- Main content: turn list + detail -->
  <div class="flex-1 flex flex-col overflow-hidden">
    {#if traceStore.loading}
      <div class="flex-1 flex items-center justify-center">
        <p class="text-[var(--text-muted)]">Loading traces...</p>
      </div>
    {:else if traceStore.error}
      <div class="flex-1 flex items-center justify-center">
        <p class="text-[var(--accent-red)]">{traceStore.error}</p>
      </div>
    {:else if traceStore.traces.length === 0}
      <div class="flex-1 flex items-center justify-center">
        <p class="text-[var(--text-muted)]">No traces for this selection</p>
      </div>
    {:else}
      <div class="flex-1 flex overflow-hidden">
        <!-- Turn list -->
        <div class="w-80 border-r border-[var(--border)] overflow-y-auto">
          {#each traceStore.allTracesSorted as trace}
                <button
                  class="w-full text-left px-3 py-2 text-sm border-b border-[var(--border)]/50 transition-colors cursor-pointer {traceStore.selectedTurnId === (trace.metadata?.turnId || trace.timestamp) ? 'bg-[var(--bg-hover)] border-l-2 border-l-[var(--accent-blue)]' : 'hover:bg-[var(--bg-hover)]'}"
                  onclick={() => {
                    const turnId = trace.metadata?.turnId || trace.timestamp
                    traceStore.selectTurn(turnId)
                    onSelectTurn?.(turnId)
                  }}
                >
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-1.5">
                      <span class="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0" style="background: {getCallTypeColor(trace.callType)}"></span>
                      <span class="font-mono text-xs" style="color: {getCallTypeColor(trace.callType)}">{trace.callType ?? 'chat'}</span>
                    </div>
                    <span class="text-xs text-[var(--text-muted)]">{formatTime(trace.timestamp)}</span>
                  </div>
                  <div class="flex items-center gap-2 mt-1">
                    <span class="text-xs text-[var(--text-secondary)]">{trace.model ?? 'unknown'}</span>
                    {#if trace.usage.inputTokens}
                      <span class="text-xs text-[var(--text-muted)]">{formatTokens(trace.usage.inputTokens)} in</span>
                    {/if}
                    {#if trace.usage.outputTokens}
                      <span class="text-xs text-[var(--text-muted)]">{formatTokens(trace.usage.outputTokens)} out</span>
                    {/if}
                  </div>
                  {#if trace.metadata?.forkId}
                    <div class="text-xs text-[var(--accent-purple)] mt-0.5 font-mono">{trace.metadata.forkId.slice(0, 8)}</div>
                  {/if}
                </button>
          {/each}
        </div>

        <!-- Turn detail -->
        <div class="flex-1 overflow-y-auto">
          {#if traceStore.selectedTrace}
            <TurnDetail trace={traceStore.selectedTrace} />
          {:else}
            <div class="flex-1 flex items-center justify-center h-full">
              <p class="text-[var(--text-muted)]">Select a turn to view details</p>
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
  </div>
</div>
