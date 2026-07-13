import type { AgentCallTrace, SessionInfo, ForkNode, SessionPage } from '../types'



class TraceStore {
  traces = $state<AgentCallTrace[]>([])
  sessions = $state<SessionInfo[]>([])
  selectedSessionId = $state<string | null>(null)
  selectedForkId = $state<string | null | undefined>(undefined)
  hiddenForkIds = $state<Set<string | null>>(new Set())
  selectedTraceId = $state<string | null>(null)
  hiddenCallTypes = $state<Set<string>>(new Set())
  loading = $state(false)
  sessionsLoading = $state(false)
  sessionsLoadingMore = $state(false)
  sessionsCursor = $state<string | null>(null)
  hasMoreSessions = $state(true)
  error = $state<string | null>(null)
  private eventSource: EventSource | null = null

  availableForks = $derived.by(() => {
    const forks = new Map<string | null, { count: number; name: string }>()
    for (const t of this.traces) {
      const forkId = t.actor.forkId
      const existing = forks.get(forkId)
      if (existing) {
        existing.count++
      } else {
        forks.set(forkId, {
          count: 1,
          name: forkId === null ? 'root' : forkId.slice(0, 8),
        })
      }
    }
    return forks
  })

  allTracesSorted = $derived.by(() => {
    let traces = [...this.traces]
    if (this.hiddenForkIds.size > 0) {
      traces = traces.filter(t => !this.hiddenForkIds.has(t.actor.forkId))
    }
    if (this.hiddenCallTypes.size > 0) {
      traces = traces.filter(t => !this.hiddenCallTypes.has(t.callType))
    }
    return traces.sort((a, b) => a.startedAt - b.startedAt)
  })

  selectedTrace = $derived.by(() => {
    if (this.selectedTraceId === null) return null
    return this.traces.find(t => t.traceId === this.selectedTraceId) ?? null
  })

  forkTree = $derived.by(() => {
    return buildForkTree(this.traces)
  })

  totalTokens = $derived.by(() => {
    let input = 0
    let output = 0
    for (const t of this.traces) {
      if (t.response.usage?.inputTokens) input += t.response.usage.inputTokens
      if (t.response.usage?.outputTokens) output += t.response.usage.outputTokens
    }
    return { input, output, total: input + output }
  })

  async fetchSessionsInitial(limit = 50) {
    this.sessionsLoading = true
    this.sessionsLoadingMore = false
    this.error = null
    this.sessions = []
    this.sessionsCursor = null
    this.hasMoreSessions = true
    try {
      const res = await fetch(`/api/sessions?limit=${limit}`)
      if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.status}`)
      const page = await res.json() as SessionPage
      this.sessions = page.items
      this.sessionsCursor = page.nextCursor
      this.hasMoreSessions = page.nextCursor !== null
    } catch (e: any) {
      this.error = e.message
      this.hasMoreSessions = false
    } finally {
      this.sessionsLoading = false
    }
  }

  async fetchMoreSessions(limit = 50) {
    if (!this.hasMoreSessions || this.sessionsLoading || this.sessionsLoadingMore) return
    if (!this.sessionsCursor) return
    this.sessionsLoadingMore = true
    this.error = null

    try {
      const cursor = encodeURIComponent(this.sessionsCursor)
      const res = await fetch(`/api/sessions?limit=${limit}&cursor=${cursor}`)
      if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.status}`)
      const page = await res.json() as SessionPage
      this.sessions = [...this.sessions, ...page.items]
      this.sessionsCursor = page.nextCursor
      this.hasMoreSessions = page.nextCursor !== null
    } catch (e: any) {
      this.error = e.message
    } finally {
      this.sessionsLoadingMore = false
    }
  }

  async selectSession(id: string) {
    this.selectedSessionId = id
    this.hiddenForkIds = new Set()
    this.selectedTraceId = null
    this.traces = []
    this.loading = true
    this.error = null
    this.disconnectSSE()

    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/traces`)
      if (!res.ok) throw new Error(`Failed to fetch traces: ${res.status}`)
      this.traces = await res.json()
      this.connectSSE(id)
    } catch (e: any) {
      this.error = e.message
    } finally {
      this.loading = false
    }
  }

  private connectSSE(sessionId: string) {
    this.disconnectSSE()
    const es = new EventSource(`/api/sessions/${encodeURIComponent(sessionId)}/stream`)
    es.onmessage = (event) => {
      try {
        const trace: AgentCallTrace = JSON.parse(event.data)
        this.traces = [...this.traces, trace]
      } catch {}
    }
    es.onerror = () => {}
    this.eventSource = es
  }

  private disconnectSSE() {
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
  }

  toggleFork(forkId: string | null) {
    const next = new Set(this.hiddenForkIds)
    if (next.has(forkId)) {
      next.delete(forkId)
    } else {
      next.add(forkId)
    }
    this.hiddenForkIds = next
    this.selectedForkId = undefined
    this.selectedTraceId = null
  }

  showAllForks() {
    if (this.hiddenForkIds.size === 0) {
      this.hiddenForkIds = new Set(this.availableForks.keys())
    } else {
      this.hiddenForkIds = new Set()
    }
    this.selectedForkId = undefined
    this.selectedTraceId = null
  }

  selectFork(forkId: string | null) {
    this.selectedForkId = forkId
    this.hiddenForkIds = new Set(
      [...this.availableForks.keys()].filter((candidate) => candidate !== forkId),
    )
    this.selectedTraceId = null
  }

  toggleCallType(type: string) {
    const next = new Set(this.hiddenCallTypes)
    if (next.has(type)) {
      next.delete(type)
    } else {
      next.add(type)
    }
    this.hiddenCallTypes = next
    this.selectedTraceId = null
  }

  showAllCallTypes() {
    if (this.hiddenCallTypes.size === 0) {
      this.hiddenCallTypes = new Set(this.callTypes)
    } else {
      this.hiddenCallTypes = new Set()
    }
    this.selectedTraceId = null
  }

  selectTrace(traceId: string | null) {
    this.selectedTraceId = traceId
  }

  clearSelection() {
    this.hiddenForkIds = new Set()
    this.hiddenCallTypes = new Set()
    this.selectedForkId = undefined
  }

  get callTypes(): string[] {
    const types = new Set<string>()
    for (const t of this.traces) {
      types.add(t.callType)
    }
    return [...types].sort()
  }

  destroy() {
    this.disconnectSSE()
  }
}

function buildForkTree(traces: AgentCallTrace[]): ForkNode[] {
  const forkMap = new Map<string | null, { count: number }>()

  for (const t of traces) {
    const key = t.actor.forkId
    if (!forkMap.has(key)) {
      forkMap.set(key, { count: 0 })
    }
    forkMap.get(key)!.count++
  }

  const nodes: ForkNode[] = []
  for (const [forkId, info] of forkMap) {
    nodes.push({
      forkId,
      name: forkId === null ? 'root' : forkId.slice(0, 8),
      mode: forkId === null ? 'root' : 'spawn',
      parentForkId: null,
      children: [],
      traceCount: info.count,
    })
  }

  return nodes
}

export const traceStore = new TraceStore()
