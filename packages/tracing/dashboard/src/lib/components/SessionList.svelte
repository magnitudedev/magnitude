<script lang="ts">
  import { onMount } from 'svelte'
  import { traceStore } from '../stores/traces.svelte'

  let { onSelect }: { onSelect: (id: string) => void } = $props()
  let sentinel = $state<HTMLDivElement | null>(null)
  let observer = $state<IntersectionObserver | null>(null)

  function canFetchMoreSessions() {
    return traceStore.hasMoreSessions && !traceStore.sessionsLoading && !traceStore.sessionsLoadingMore
  }

  onMount(() => {
    observer = new IntersectionObserver((entries) => {
      const visible = entries.some((e) => e.isIntersecting)
      if (visible && canFetchMoreSessions()) {
        void traceStore.fetchMoreSessions()
      }
    }, { rootMargin: '400px 0px' })

    void traceStore.fetchSessionsInitial()

    return () => observer?.disconnect()
  })

  $effect(() => {
    if (!observer || !sentinel) return
    observer.observe(sentinel)
    return () => {
      if (sentinel) observer?.unobserve(sentinel)
    }
  })

  function formatDate(ts: string): string {
    try {
      const d = new Date(ts)
      return d.toLocaleString()
    } catch {
      return ts
    }
  }
</script>

<div class="p-6">
  <h1 class="text-2xl font-semibold mb-6 text-[var(--text-primary)]">Sessions</h1>

  {#if traceStore.sessionsLoading}
    <p class="text-[var(--text-muted)]">Loading sessions...</p>
  {:else if traceStore.error}
    <p class="text-[var(--accent-red)]">{traceStore.error}</p>
  {:else if traceStore.sessions.length === 0}
    <div class="text-[var(--text-muted)] text-center py-12">
      <p class="text-lg mb-2">No trace sessions found</p>
      <p class="text-sm">Traces will appear here once Magnitude records LLM calls to ~/.magnitude/traces/</p>
    </div>
  {:else}
    <div class="space-y-2">
      {#each traceStore.sessions as session}
        <button
          class="w-full text-left p-4 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] hover:bg-[var(--bg-hover)] hover:border-[var(--accent-blue)]/30 transition-colors cursor-pointer"
          onclick={() => onSelect(session.id)}
        >
          <div class="flex items-center justify-between">
            <div>
              {#if session.meta?.chatName}
                <span class="text-sm text-[var(--text-primary)]">{session.meta.chatName}</span>
              {:else}
                <span class="text-sm text-[var(--text-muted)]">New Chat</span>
              {/if}
              <span class="ml-3 text-sm text-[var(--text-muted)]">{formatDate(session.timestamp)}</span>
            </div>
            <div class="flex items-center gap-4 text-sm text-[var(--text-secondary)]">
              {#if session.traceCount !== undefined}
                <span>{session.traceCount} traces</span>
              {/if}
            </div>
          </div>
        </button>
      {/each}
    </div>
    <div bind:this={sentinel} class="h-4"></div>
    {#if traceStore.sessionsLoadingMore}
      <div class="mt-3 flex items-center gap-2 text-sm text-[var(--text-muted)]">
        <span class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-[var(--border)] border-t-[var(--accent-blue)]"></span>
        <span>Loading more sessions…</span>
      </div>
    {/if}
  {/if}
</div>
