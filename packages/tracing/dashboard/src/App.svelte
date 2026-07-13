<script lang="ts">
  import { onMount } from 'svelte'
  import SessionList from './lib/components/SessionList.svelte'
  import SessionDetail from './lib/components/SessionDetail.svelte'

  let selectedSessionId = $state<string | null>(null)
  let selectedTraceId = $state<string | null>(null)

  function parsePath(pathname: string): { sessionId: string | null; traceId: string | null } {
    if (pathname === '/') return { sessionId: null, traceId: null }
    const traceMatch = pathname.match(/^\/session\/([^/]+)\/trace\/([^/]+)$/)
    if (traceMatch) {
      return {
        sessionId: decodeURIComponent(traceMatch[1]),
        traceId: decodeURIComponent(traceMatch[2]),
      }
    }
    const sessionMatch = pathname.match(/^\/session\/([^/]+)$/)
    if (sessionMatch) {
      return {
        sessionId: decodeURIComponent(sessionMatch[1]),
        traceId: null,
      }
    }
    return { sessionId: null, traceId: null }
  }

  function applyRoute(pathname: string) {
    const route = parsePath(pathname)
    selectedSessionId = route.sessionId
    selectedTraceId = route.traceId
  }

  function navigate(path: string, replace = false) {
    if (replace) history.replaceState(null, '', path)
    else history.pushState(null, '', path)
    applyRoute(path)
  }

  function handleSelectSession(id: string) {
    navigate(`/session/${encodeURIComponent(id)}`)
  }

  function handleTraceSelection(traceId: string | null, replace = false) {
    if (!selectedSessionId) return
    if (!traceId) {
      navigate(`/session/${encodeURIComponent(selectedSessionId)}`, replace)
      return
    }
    navigate(`/session/${encodeURIComponent(selectedSessionId)}/trace/${encodeURIComponent(traceId)}`, replace)
  }

  function handleBack() {
    navigate('/')
  }

  onMount(() => {
    applyRoute(window.location.pathname)
    const onPopState = () => applyRoute(window.location.pathname)
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  })
</script>

<div class="min-h-screen bg-[var(--bg-primary)]">
  {#if !selectedSessionId}
    <SessionList onSelect={handleSelectSession} />
  {:else}
    <SessionDetail
      sessionId={selectedSessionId}
      selectedTraceIdFromRoute={selectedTraceId}
      onSelectTrace={handleTraceSelection}
      onBack={handleBack}
    />
  {/if}
</div>
