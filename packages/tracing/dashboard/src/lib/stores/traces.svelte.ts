import type { AgentTrace, SessionInfo, ForkNode, SessionPage } from '../types'



class TraceStore {
  traces = $state<AgentTrace[]>([])
  sessions = $state<SessionInfo[]>([])
  selectedSessionId = $state<string | null>(null)
  hiddenForkIds = $state<Set<string | null>>(new Set())
  selectedTurnId = $state<string | null>(null)
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
      const forkId = t.metadata?.forkId ?? null
      const existing = forks.get(forkId)
      if (existing) {
        existing.count++
        if (!existing.name && t.metadata?.callType === 'chat') existing.name = t.metadata.forkName
      } else {
        forks.set(forkId, {
          count: 1,
          name: forkId === null ? 'root' : (t.metadata?.callType === 'chat' ? t.metadata.forkName : forkId.slice(0, 8)),
        })
      }
    }
    return forks
  })

  allTracesSorted = $derived.by(() => {
    let traces = [...this.traces]
    if (this.hiddenForkIds.size > 0) {
      traces = traces.filter(t => !this.hiddenForkIds.has(t.metadata?.forkId ?? null))
    }
    if (this.hiddenCallTypes.size > 0) {
      traces = traces.filter(t => !this.hiddenCallTypes.has(t.callType ?? 'chat'))
    }
    return traces.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
  })

  groupedByChain = $derived.by(() => {
    const groups = new Map<string, AgentTrace[]>()
    for (const trace of this.allTracesSorted.filter(t => !t.callType || t.callType === 'chat')) {
      const chainId = trace.metadata?.chainId ?? ''
      const existing = groups.get(chainId)
      if (existing) {
        existing.push(trace)
      } else {
        groups.set(chainId, [trace])
      }
    }
    return groups
  })

  selectedTrace = $derived.by(() => {
    if (this.selectedTurnId === null) return null
    return this.traces.find(t => (t.metadata?.turnId || t.timestamp) === this.selectedTurnId) ?? null
  })

  forkTree = $derived.by(() => {
    return buildForkTree(this.traces)
  })

  totalTokens = $derived.by(() => {
    let input = 0
    let output = 0
    for (const t of this.traces) {
      if (t.usage.inputTokens) input += t.usage.inputTokens
      if (t.usage.outputTokens) output += t.usage.outputTokens
    }
    return { input, output, total: input + output }
  })

  totalCost = $derived.by(() => {
    let cost = 0
    for (const t of this.traces) {
      if (t.usage.totalCost) cost += t.usage.totalCost
    }
    return cost
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
    this.selectedTurnId = null
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
        const trace: AgentTrace = JSON.parse(event.data)
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
    this.selectedTurnId = null
  }

  showAllForks() {
    if (this.hiddenForkIds.size === 0) {
      // All shown → hide all
      this.hiddenForkIds = new Set(this.availableForks.keys())
    } else {
      this.hiddenForkIds = new Set()
    }
    this.selectedTurnId = null
  }

  toggleCallType(type: string) {
    const next = new Set(this.hiddenCallTypes)
    if (next.has(type)) {
      next.delete(type)
    } else {
      next.add(type)
    }
    this.hiddenCallTypes = next
    this.selectedTurnId = null
  }

  showAllCallTypes() {
    if (this.hiddenCallTypes.size === 0) {
      // All shown → hide all
      this.hiddenCallTypes = new Set(this.callTypes)
    } else {
      this.hiddenCallTypes = new Set()
    }
    this.selectedTurnId = null
  }

  selectTurn(turnId: string | null) {
    this.selectedTurnId = turnId
  }

  clearSelection() {
    this.hiddenForkIds = new Set()
    this.hiddenCallTypes = new Set()
  }

  get callTypes(): string[] {
    const types = new Set<string>()
    for (const t of this.traces) {
      types.add(t.callType ?? 'chat')
    }
    return [...types].sort()
  }

  destroy() {
    this.disconnectSSE()
  }
}

function buildForkTree(traces: AgentTrace[]): ForkNode[] {
  const forkMap = new Map<string | null, { name: string; mode: string; parentForkId: string | null; count: number }>()

  for (const t of traces) {
    const key = t.metadata?.forkId ?? null
    if (!forkMap.has(key)) {
      forkMap.set(key, {
        name: key === null ? 'root' : (t.metadata?.callType === 'chat' ? t.metadata.forkName : key),
        mode: key === null ? 'root' : (t.slot === 'secondary' ? 'spawn' : 'clone'),
        parentForkId: null,
        count: 0,
      })
    }
    forkMap.get(key)!.count++
  }

  const nodes: ForkNode[] = []
  for (const [forkId, info] of forkMap) {
    nodes.push({
      forkId: forkId,
      name: info.name,
      mode: info.mode as any,
      parentForkId: info.parentForkId,
      children: [],
      traceCount: info.count,
    })
  }

  return nodes
}

export const traceStore = new TraceStore()
