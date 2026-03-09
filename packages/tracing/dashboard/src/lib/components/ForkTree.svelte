<script lang="ts">
  import { traceStore } from '../stores/traces.svelte'
  import type { ForkNode } from '../types'

  const modeColors: Record<string, string> = {
    root: 'var(--accent-blue)',
    clone: 'var(--accent-purple)',
    spawn: 'var(--accent-green)',
  }

  function isSelected(forkId: string | null): boolean {
    if (traceStore.selectedForkId === undefined) return false
    return traceStore.selectedForkId === forkId
  }
</script>

<div class="p-3">
  <div class="flex items-center justify-between mb-3">
    <h3 class="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">Forks</h3>
    {#if traceStore.selectedForkId !== undefined}
      <button
        class="text-xs text-[var(--accent-blue)] hover:underline cursor-pointer"
        onclick={() => traceStore.clearSelection()}
      >
        Show all
      </button>
    {/if}
  </div>

  <div class="space-y-1">
    {#each traceStore.forkTree as node}
      <button
        class="w-full text-left px-3 py-2 rounded text-sm transition-colors cursor-pointer {isSelected(node.forkId) ? 'bg-[var(--bg-hover)] border border-[var(--accent-blue)]/40' : 'hover:bg-[var(--bg-hover)] border border-transparent'}"
        onclick={() => traceStore.selectFork(node.forkId)}
      >
        <div class="flex items-center gap-2">
          <span
            class="w-2 h-2 rounded-full flex-shrink-0"
            style="background: {modeColors[node.mode] || 'var(--text-muted)'}"
          ></span>
          <span class="font-mono text-xs truncate" title={node.forkId ?? 'root'}>{node.name}</span>
          {#if node.forkId}
            <span class="text-[10px] text-[var(--text-muted)] font-mono truncate max-w-[80px]" title={node.forkId}>{node.forkId.slice(0, 8)}</span>
          {/if}
          <span class="ml-auto text-xs text-[var(--text-muted)]">{node.traceCount}</span>
        </div>
      </button>
    {/each}
  </div>
</div>
