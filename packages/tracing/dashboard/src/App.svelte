<script lang="ts">
  import { onMount } from 'svelte'
  import SessionList from './lib/components/SessionList.svelte'
  import SessionDetail from './lib/components/SessionDetail.svelte'

  let selectedSessionId = $state<string | null>(null)
  let selectedTurnId = $state<string | null>(null)

  function parsePath(pathname: string): { sessionId: string | null; turnId: string | null } {
    if (pathname === '/') return { sessionId: null, turnId: null }
    const turnMatch = pathname.match(/^\/session\/([^/]+)\/turn\/([^/]+)$/)
    if (turnMatch) {
      return {
        sessionId: decodeURIComponent(turnMatch[1]),
        turnId: decodeURIComponent(turnMatch[2]),
      }
    }
    const sessionMatch = pathname.match(/^\/session\/([^/]+)$/)
    if (sessionMatch) {
      return {
        sessionId: decodeURIComponent(sessionMatch[1]),
        turnId: null,
      }
    }
    return { sessionId: null, turnId: null }
  }

  function applyRoute(pathname: string) {
    const route = parsePath(pathname)
    selectedSessionId = route.sessionId
    selectedTurnId = route.turnId
  }

  function navigate(path: string, replace = false) {
    if (replace) history.replaceState(null, '', path)
    else history.pushState(null, '', path)
    applyRoute(path)
  }

  function handleSelectSession(id: string) {
    navigate(`/session/${encodeURIComponent(id)}`)
  }

  function handleTurnSelection(turnId: string | null, replace = false) {
    if (!selectedSessionId) return
    if (!turnId) {
      navigate(`/session/${encodeURIComponent(selectedSessionId)}`, replace)
      return
    }
    navigate(`/session/${encodeURIComponent(selectedSessionId)}/turn/${encodeURIComponent(turnId)}`, replace)
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
      selectedTurnIdFromRoute={selectedTurnId}
      onSelectTurn={handleTurnSelection}
      onBack={handleBack}
    />
  {/if}
</div>
