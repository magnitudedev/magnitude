<script lang="ts">
  import { onMount } from 'svelte'
  import MemoryAtlasCanvas from './MemoryAtlasCanvas.svelte'
  import type {
    AcnDisplayViewIntrospection,
    AcnInfo,
    AcnSession,
    AcnSessionIntrospection,
    AddressedAtlasNode,
    AddressedAtlasResident,
    AddressedAtlasSegment,
    AddressedPin,
    KillAllAcnResult,
    ProjectionIntrospection,
    RpcTraceSummary,
  } from './lib/types'

  interface ClientSegmentPin {
    readonly node: AddressedAtlasSegment
  }

  interface ClientSummary {
    readonly id: string
    readonly view: AcnDisplayViewIntrospection | null
    readonly shapeLabel: string
    readonly subscriberCount: number
    readonly lastActivityAt: number | null
    readonly pinnedSegments: readonly ClientSegmentPin[]
    readonly pinnedBytes: number
  }

  interface MemoryStats {
    readonly addressedRootCount: number
    readonly loadedEntries: number
    readonly offloadedEntries: number
    readonly pinnedEntries: number
    readonly producerPinnedEntries: number
    readonly projectionStateBytes: number
    readonly addressedBytes: number
    readonly knownBytes: number
  }

  type WorkspaceTab = 'atlas' | 'projections' | 'rpc'

  type InspectorSelection =
    | { readonly kind: 'session' }
    | { readonly kind: 'client'; readonly clientId: string }
    | { readonly kind: 'projection'; readonly projectionName: string }
    | { readonly kind: 'addressed'; readonly node: AddressedAtlasNode }

  let acns = $state<AcnInfo[]>([])
  let sessions = $state<AcnSession[]>([])
  let selectedVersion = $state<string | null>(null)
  let selectedSessionId = $state<string | null>(null)
  let selectedIntrospection = $state<AcnSessionIntrospection | null>(null)
  let rpcTraces = $state<RpcTraceSummary[]>([])
  let activeTab = $state<WorkspaceTab>('atlas')
  let inspectorSelection = $state<InspectorSelection>({ kind: 'session' })
  let selectedProjectionName = $state<string | null>(null)
  let hoveredClientId = $state<string | null>(null)
  let selectedClientId = $state<string | null>(null)
  let loadingAcns = $state(false)
  let loadingSessions = $state(false)
  let killingAcns = $state(false)
  let error = $state<string | null>(null)
  let notice = $state<string | null>(null)
  let rpcTraceError = $state<string | null>(null)
  let streamState = $state<'idle' | 'connecting' | 'live' | 'error'>('idle')

  const selectedAcn = $derived(acns.find((acn) => acn.version === selectedVersion) ?? null)
  const selectedSession = $derived(sessions.find((session) => session.sessionId === selectedSessionId) ?? null)
  const agentIntrospection = $derived(selectedIntrospection?.introspection ?? null)
  const projections = $derived(agentIntrospection?.projections ?? [])
  const addressedAtlas = $derived(agentIntrospection?.addressedAtlas ?? [])
  const displayViews = $derived(selectedIntrospection?.displayViews ?? [])
  const clients = $derived(buildClientSummaries(displayViews, addressedAtlas))
  const activeClientId = $derived(hoveredClientId ?? selectedClientId)
  const selectedInspectorProjection = $derived(selectedProjectionForInspector(inspectorSelection, projections))
  const selectedInspectorClient = $derived(selectedClientForInspector(inspectorSelection, clients))
  const selectedAddressedNode = $derived(inspectorSelection.kind === 'addressed' ? inspectorSelection.node : null)
  const selectedProjection = $derived(
    projections.find((projection) => projection.name === selectedProjectionName) ?? projections[0] ?? null,
  )
  const displayViewSubscriberCount = $derived(
    displayViews.reduce((total, view) => total + view.subscriberCount, 0),
  )
  const rpcCommands = $derived(rpcTraces.filter((trace) => trace.kind === 'command'))
  const rpcStreams = $derived(rpcTraces.filter((trace) => trace.kind === 'stream'))
  const memoryStats = $derived(buildMemoryStats(projections, addressedAtlas))

  function formatTime(timestamp: number | null | undefined): string {
    if (!timestamp) return 'never'
    return new Date(timestamp).toLocaleTimeString()
  }

  function formatBytes(bytes: number): string {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${bytes} B`
  }

  function shortId(value: string): string {
    return value.length > 12 ? value.slice(0, 12) : value
  }

  function durationLabel(trace: RpcTraceSummary): string {
    const startedAt = new Date(trace.startedAt).getTime()
    const duration = trace.isRunning && Number.isFinite(startedAt)
      ? Date.now() - startedAt
      : trace.durationMs
    if (duration >= 1000) return `${(duration / 1000).toFixed(1)}s`
    return `${Math.max(0, Math.round(duration))}ms`
  }

  function projectionLabel(projection: ProjectionIntrospection): string {
    return projection.name.replace(/Projection$/, '')
  }

  function projectionSummary(projection: ProjectionIntrospection): string {
    if (projection.summary?.label) return projection.summary.label
    const state = projection.state
    if (state == null) return 'empty'
    if (typeof state !== 'object') return typeof state
    if (Array.isArray(state)) return `${state.length} items`
    return `${Object.keys(state as Record<string, unknown>).length} keys`
  }

  function segmentRange(segment: AddressedAtlasSegment): string {
    if (segment.itemCount <= 0) return `${segment.startOffset}`
    const end = segment.startOffset + segment.itemCount - 1
    return segment.startOffset === end ? `${segment.startOffset}` : `${segment.startOffset}-${end}`
  }

  function segmentStateLabel(segment: AddressedAtlasSegment): string {
    const state = segment.residency === 'resident' ? 'loaded' : 'offloaded'
    const dirty = segment.dirty ? ' dirty' : ''
    const pinned = segment.pins.length > 0 ? ` pinned ${segment.pins.length}` : ''
    return `${state}${dirty}${pinned}`
  }

  function segmentHiddenItemCount(segment: AddressedAtlasSegment): number {
    return Math.max(0, segment.itemCount - segment.itemIdsSample.length)
  }

  function displayViewShapeSummary(view: AcnDisplayViewIntrospection): string {
    if (!view.shape) return 'shape not set'
    const timelines = Object.entries(view.shape.timelines)
    if (timelines.length === 0) return 'no timelines'
    return timelines.map(([timeline, window]) => {
      const label = window.kind === 'tail'
        ? `tail ${window.limit}`
        : `${window.start}-${window.start + window.limit}`
      return `${timeline}: ${label}${window.live ? ' live' : ''}`
    }).join(' · ')
  }

  function pinClientId(pin: AddressedPin): string | null {
    if (pin.kind !== 'display-view') return null
    return pin.viewId ?? pin.owner.replace(/^display-view:/, '')
  }

  function displayPins(node: AddressedAtlasSegment | AddressedAtlasResident): readonly AddressedPin[] {
    return node.pins.filter((pin) => pin.kind === 'display-view')
  }

  function producerPins(node: AddressedAtlasSegment | AddressedAtlasResident): readonly AddressedPin[] {
    return node.pins.filter((pin) => pin.kind === 'display-producer')
  }

  function segmentEstimatedBytes(segment: AddressedAtlasSegment): number {
    return segment.estimatedBytes ?? segment.estimatedResidentBytes ?? segment.estimatedStoredBytes ?? 0
  }

  function addressedNodeTitle(node: AddressedAtlasNode): string {
    return node.path.join(' / ')
  }

  function addressedNodeSize(node: AddressedAtlasNode): string {
    return node.bytes > 0 ? formatBytes(node.bytes) : 'unknown'
  }

  function walkAddressedNodes(
    nodes: readonly AddressedAtlasNode[],
    visit: (node: AddressedAtlasNode) => void,
  ): void {
    for (const node of nodes) {
      visit(node)
      if (node.kind === 'group') walkAddressedNodes(node.children, visit)
    }
  }

  function addressedSegments(nodes: readonly AddressedAtlasNode[]): AddressedAtlasSegment[] {
    const segments: AddressedAtlasSegment[] = []
    walkAddressedNodes(nodes, (node) => {
      if (node.kind === 'segment') segments.push(node)
    })
    return segments
  }

  function buildMemoryStats(
    currentProjections: readonly ProjectionIntrospection[],
    roots: readonly AddressedAtlasNode[],
  ): MemoryStats {
    let producerPinnedEntries = 0
    const projectionStateBytes = currentProjections.reduce(
      (total, projection) => total + (projection.summary?.estimatedBytes ?? 0),
      0,
    )
    const rootMetrics = roots.reduce((metrics, root) => ({
      bytes: metrics.bytes + root.bytes,
      residentEntryCount: metrics.residentEntryCount + root.residentEntryCount,
      offloadedEntryCount: metrics.offloadedEntryCount + root.offloadedEntryCount,
      pinnedEntryCount: metrics.pinnedEntryCount + root.pinnedEntryCount,
    }), {
      bytes: 0,
      residentEntryCount: 0,
      offloadedEntryCount: 0,
      pinnedEntryCount: 0,
    })

    for (const segment of addressedSegments(roots)) {
      if (producerPins(segment).length > 0) producerPinnedEntries += 1
    }

    return {
      addressedRootCount: roots.length,
      loadedEntries: rootMetrics.residentEntryCount,
      offloadedEntries: rootMetrics.offloadedEntryCount,
      pinnedEntries: rootMetrics.pinnedEntryCount,
      producerPinnedEntries,
      projectionStateBytes,
      addressedBytes: rootMetrics.bytes,
      knownBytes: projectionStateBytes + rootMetrics.bytes,
    }
  }

  function buildClientSummaries(
    views: readonly AcnDisplayViewIntrospection[],
    roots: readonly AddressedAtlasNode[],
  ): ClientSummary[] {
    const summaries = new Map<string, {
      view: AcnDisplayViewIntrospection | null
      pins: ClientSegmentPin[]
      pinnedBytes: number
      seenSegments: Set<string>
    }>()

    const ensure = (id: string) => {
      const existing = summaries.get(id)
      if (existing) return existing
      const created = {
        view: null,
        pins: [],
        pinnedBytes: 0,
        seenSegments: new Set<string>(),
      }
      summaries.set(id, created)
      return created
    }

    for (const view of views) {
      ensure(view.viewId).view = view
    }

    for (const segment of addressedSegments(roots)) {
      for (const pin of displayPins(segment)) {
        const id = pinClientId(pin)
        if (!id) continue
        const summary = ensure(id)
        const key = `${segment.namespace}\u0000${segment.forkId ?? 'root'}\u0000${segment.address}`
        if (summary.seenSegments.has(key)) continue
        summary.seenSegments.add(key)
        summary.pins.push({ node: segment })
        summary.pinnedBytes += segmentEstimatedBytes(segment)
      }
    }

    return [...summaries.entries()]
      .map(([id, summary]) => ({
        id,
        view: summary.view,
        shapeLabel: summary.view ? displayViewShapeSummary(summary.view) : 'pinned view',
        subscriberCount: summary.view?.subscriberCount ?? 0,
        lastActivityAt: summary.view?.lastActivityAt ?? null,
        pinnedSegments: summary.pins,
        pinnedBytes: summary.pinnedBytes,
      }))
      .sort((left, right) => left.id.localeCompare(right.id))
  }

  function selectedProjectionForInspector(
    selection: InspectorSelection,
    currentProjections: readonly ProjectionIntrospection[],
  ): ProjectionIntrospection | null {
    return selection.kind === 'projection'
      ? currentProjections.find((projection) => projection.name === selection.projectionName) ?? null
      : null
  }

  function selectedClientForInspector(
    selection: InspectorSelection,
    currentClients: readonly ClientSummary[],
  ): ClientSummary | null {
    return selection.kind === 'client'
      ? currentClients.find((client) => client.id === selection.clientId) ?? null
      : null
  }

  async function fetchAcns() {
    loadingAcns = true
    error = null
    try {
      const response = await fetch('/api/acns')
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
      const payload = await response.json() as { acns: AcnInfo[] }
      acns = payload.acns
      if (!selectedVersion || !payload.acns.some((acn) => acn.version === selectedVersion)) {
        selectedVersion = payload.acns.find((acn) => acn.introspection.ok)?.version ?? payload.acns[0]?.version ?? null
      }
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught)
    } finally {
      loadingAcns = false
    }
  }

  async function fetchSessions() {
    if (!selectedVersion) {
      sessions = []
      selectedSessionId = null
      return
    }

    loadingSessions = true
    error = null
    try {
      const response = await fetch(`/api/acns/${encodeURIComponent(selectedVersion)}/sessions`)
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
      const payload = await response.json() as { sessions: AcnSession[] }
      sessions = payload.sessions
      if (!selectedSessionId || !payload.sessions.some((session) => session.sessionId === selectedSessionId)) {
        selectedSessionId = payload.sessions[0]?.sessionId ?? null
      }
    } catch (caught) {
      sessions = []
      selectedSessionId = null
      selectedIntrospection = null
      error = caught instanceof Error ? caught.message : String(caught)
    } finally {
      loadingSessions = false
    }
  }

  async function fetchRpcTraces() {
    rpcTraceError = null
    try {
      const response = await fetch('/api/rpc-traces')
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
      const payload = await response.json() as { traces: RpcTraceSummary[], error?: string }
      rpcTraces = payload.traces
      rpcTraceError = payload.error ?? null
    } catch (caught) {
      rpcTraces = []
      rpcTraceError = caught instanceof Error ? caught.message : String(caught)
    }
  }

  function killSummary(results: KillAllAcnResult[]): string {
    const killed = results.filter((result) => result.status === 'killed').length
    const stale = results.filter((result) => result.status === 'stale').length
    const failed = results.filter((result) => result.status === 'failed').length
    if (results.length === 0) return 'No registered ACNs found'
    return `Killed ${killed} ACN${killed === 1 ? '' : 's'}${stale ? `, removed ${stale} stale` : ''}${failed ? `, ${failed} failed` : ''}`
  }

  async function killAllAcns() {
    if (killingAcns) return
    const ok = window.confirm('Kill all registered ACN processes?')
    if (!ok) return

    killingAcns = true
    error = null
    notice = null
    try {
      const response = await fetch('/api/acns/kill-all', { method: 'POST' })
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
      const payload = await response.json() as { results: KillAllAcnResult[] }
      notice = killSummary(payload.results)
      await new Promise((resolve) => setTimeout(resolve, 500))
      await fetchAcns()
      sessions = []
      selectedSessionId = null
      selectedIntrospection = null
    } catch (caught) {
      error = caught instanceof Error ? caught.message : String(caught)
    } finally {
      killingAcns = false
    }
  }

  onMount(() => {
    void fetchAcns()
    void fetchRpcTraces()
    const interval = setInterval(() => {
      void fetchAcns()
      void fetchRpcTraces()
    }, 3000)
    return () => clearInterval(interval)
  })

  $effect(() => {
    selectedVersion
    void fetchSessions()
  })

  $effect(() => {
    if (!selectedVersion || !selectedSessionId) {
      streamState = 'idle'
      selectedIntrospection = null
      return
    }

    streamState = 'connecting'
    inspectorSelection = { kind: 'session' }
    selectedProjectionName = null
    selectedClientId = null
    hoveredClientId = null
    const source = new EventSource(
      `/api/acns/${encodeURIComponent(selectedVersion)}/sessions/${encodeURIComponent(selectedSessionId)}/stream`,
    )

    source.onopen = () => {
      streamState = 'live'
    }
    source.onmessage = (event) => {
      selectedIntrospection = JSON.parse(event.data) as AcnSessionIntrospection
    }
    source.onerror = () => {
      streamState = 'error'
      source.close()
    }

    return () => source.close()
  })

  $effect(() => {
    if (projections.length === 0) {
      selectedProjectionName = null
      return
    }

    if (!selectedProjectionName || !projections.some((projection) => projection.name === selectedProjectionName)) {
      selectedProjectionName = projections[0].name
    }
  })
</script>

<main>
  <header class="topbar">
    <div class="brand">
      <span class="mark"></span>
      <div>
        <h1>ACN Dashboard</h1>
        <p>{selectedAcn?.registration.url ?? 'No ACN selected'}</p>
      </div>
    </div>

    <div class="controls">
      <select bind:value={selectedVersion} aria-label="ACN version">
        {#each acns as acn}
          <option value={acn.version}>{acn.version} · pid {acn.registration.pid}</option>
        {/each}
      </select>
      <button class="icon-button" onclick={() => void fetchAcns()} aria-label="Refresh ACNs" title="Refresh ACNs">↻</button>
      <button class="danger" disabled={killingAcns} onclick={() => void killAllAcns()}>
        {killingAcns ? 'Killing' : 'Kill ACNs'}
      </button>
      <span class="status" data-state={streamState}>{streamState}</span>
    </div>
  </header>

  {#if error}
    <div class="banner">{error}</div>
  {/if}
  {#if notice}
    <div class="notice">{notice}</div>
  {/if}

  <section class="workspace">
    <aside class="sessions">
      <div class="pane-head">
        <h2>Sessions</h2>
        <span>{loadingSessions ? 'loading' : sessions.length}</span>
      </div>
      {#if loadingAcns}
        <div class="empty">Scanning ACNs</div>
      {:else if !selectedAcn}
        <div class="empty">No registered ACNs</div>
      {:else if !selectedAcn.introspection.ok}
        <div class="empty">Introspection unavailable for {selectedAcn.version}</div>
      {:else if sessions.length === 0}
        <div class="empty">No live sessions</div>
      {:else}
        <div class="session-list">
          {#each sessions as session}
            <button
              class:selected={session.sessionId === selectedSessionId}
              onclick={() => selectedSessionId = session.sessionId}
            >
              <strong>{session.title}</strong>
              <span>{shortId(session.sessionId)}</span>
              <small>{session.cwd}</small>
            </button>
          {/each}
        </div>
      {/if}
    </aside>

    <aside class="clients">
      <div class="pane-head">
        <h2>Clients</h2>
        <span>{clients.length}</span>
      </div>
      {#if !selectedIntrospection}
        <div class="empty">Select a live session</div>
      {:else if clients.length === 0}
        <div class="empty">No display clients</div>
      {:else}
        <div class="client-list">
          {#each clients as client}
            <button
              class:active={activeClientId === client.id}
              class:locked={selectedClientId === client.id}
              onmouseenter={() => hoveredClientId = client.id}
              onmouseleave={() => hoveredClientId = null}
              onclick={() => {
                selectedClientId = selectedClientId === client.id ? null : client.id
                inspectorSelection = { kind: 'client', clientId: client.id }
                activeTab = 'atlas'
              }}
            >
              <strong>{client.id}</strong>
              <span>{client.shapeLabel}</span>
              <small>{client.pinnedSegments.length} pinned · {formatBytes(client.pinnedBytes)} · {client.subscriberCount} stream{client.subscriberCount === 1 ? '' : 's'}</small>
            </button>
          {/each}
        </div>
      {/if}
    </aside>

    <section class="dashboard">
      <div class="summary-row">
        <div class="metric">
          <span>projections</span>
          <strong>{projections.length}</strong>
        </div>
        <div class="metric">
          <span>clients</span>
          <strong>{clients.length}</strong>
        </div>
        <div class="metric">
          <span>atlas bytes</span>
          <strong>{formatBytes(memoryStats.knownBytes)}</strong>
        </div>
        <div class="metric">
          <span>last command</span>
          <strong>{formatTime(selectedIntrospection?.activity.lastCommandAt)}</strong>
        </div>
        <div class="metric">
          <span>tokens</span>
          <strong>{agentIntrospection?.contextUsage?.currentTokens ?? 0}</strong>
        </div>
      </div>

      <nav class="tabs" aria-label="Dashboard views">
        <button class:active={activeTab === 'atlas'} onclick={() => activeTab = 'atlas'}>Memory Atlas</button>
        <button class:active={activeTab === 'projections'} onclick={() => activeTab = 'projections'}>Projections</button>
        <button class:active={activeTab === 'rpc'} onclick={() => activeTab = 'rpc'}>RPC / Activity</button>
      </nav>

      {#if activeTab === 'atlas'}
        <div class="atlas-layout">
          <section class="atlas-panel">
            <div class="map-head">
              <div>
                <h2>Memory / Pin Atlas</h2>
                <span>{memoryStats.addressedRootCount} addressed roots · {formatBytes(memoryStats.projectionStateBytes)} state · {formatBytes(memoryStats.addressedBytes)} addressed · {memoryStats.loadedEntries} loaded · {memoryStats.offloadedEntries} offloaded</span>
              </div>
              <em>{activeClientId ? `highlighting ${activeClientId}` : 'all pins'}</em>
            </div>
            {#if projections.length === 0}
              <div class="empty">Waiting for introspection</div>
            {:else}
              <MemoryAtlasCanvas
                {projections}
                {addressedAtlas}
                {activeClientId}
                onSelectNode={(node) => {
                  inspectorSelection = { kind: 'addressed', node }
                }}
                onSelectProjection={(projectionName) => {
                  inspectorSelection = { kind: 'projection', projectionName }
                }}
              />
            {/if}
          </section>

          <aside class="atlas-inspector">
            <div class="pane-head">
              <h2>Inspector</h2>
              <span>{selectedSession ? shortId(selectedSession.sessionId) : '-'}</span>
            </div>
            {#if selectedAddressedNode}
              <h3>{addressedNodeTitle(selectedAddressedNode)}</h3>
              <dl>
                <dt>size</dt>
                <dd>{addressedNodeSize(selectedAddressedNode)}</dd>
                <dt>loaded</dt>
                <dd>{selectedAddressedNode.residentEntryCount}</dd>
                <dt>offloaded</dt>
                <dd>{selectedAddressedNode.offloadedEntryCount}</dd>
                <dt>pinned</dt>
                <dd>{selectedAddressedNode.pinnedEntryCount}</dd>
                <dt>dirty</dt>
                <dd>{selectedAddressedNode.dirtyEntryCount}</dd>
              </dl>
              {#if selectedAddressedNode.kind === 'segment'}
                <dl>
                  <dt>fork</dt>
                  <dd>{selectedAddressedNode.forkId ? shortId(selectedAddressedNode.forkId) : 'root'}</dd>
                  <dt>segment</dt>
                  <dd>{selectedAddressedNode.logicalSegmentId}</dd>
                  <dt>range</dt>
                  <dd>{segmentRange(selectedAddressedNode)}</dd>
                  <dt>state</dt>
                  <dd>{segmentStateLabel(selectedAddressedNode)}</dd>
                  <dt>resident bytes</dt>
                  <dd>{selectedAddressedNode.estimatedResidentBytes === null ? 'not resident' : formatBytes(selectedAddressedNode.estimatedResidentBytes)}</dd>
                  <dt>stored bytes</dt>
                  <dd>{selectedAddressedNode.estimatedStoredBytes === null ? 'unknown' : formatBytes(selectedAddressedNode.estimatedStoredBytes)}</dd>
                  <dt>address</dt>
                  <dd>{selectedAddressedNode.address}</dd>
                </dl>
                <section class="activity-section">
                  <h3>Consumer Pins</h3>
                  <div class="activity-group">
                    {#if displayPins(selectedAddressedNode).length === 0}
                      <small>none</small>
                    {:else}
                      {#each displayPins(selectedAddressedNode) as pin}
                        <button
                          class="pin-row consumer"
                          onmouseenter={() => hoveredClientId = pinClientId(pin)}
                          onmouseleave={() => hoveredClientId = null}
                          onclick={() => {
                            const clientId = pinClientId(pin)
                            if (clientId) {
                              selectedClientId = clientId
                              inspectorSelection = { kind: 'client', clientId }
                            }
                          }}
                        >
                          {pinClientId(pin) ?? pin.owner}
                        </button>
                      {/each}
                    {/if}
                  </div>
                </section>
                <section class="activity-section">
                  <h3>Producer Pins</h3>
                  <div class="activity-group">
                    {#if producerPins(selectedAddressedNode).length === 0}
                      <small>none</small>
                    {:else}
                      {#each producerPins(selectedAddressedNode) as pin}
                        <span class="pin-row producer">{pin.owner}</span>
                      {/each}
                    {/if}
                  </div>
                </section>
                <section class="activity-section">
                  <h3>Item IDs</h3>
                  <div class="item-list">
                    {#each selectedAddressedNode.itemIdsSample as itemId}
                      <span>{itemId}</span>
                    {/each}
                    {#if segmentHiddenItemCount(selectedAddressedNode) > 0}
                      <span>+{segmentHiddenItemCount(selectedAddressedNode)} more</span>
                    {/if}
                  </div>
                </section>
              {:else if selectedAddressedNode.kind === 'resident'}
                <dl>
                  <dt>resident bytes</dt>
                  <dd>{formatBytes(selectedAddressedNode.estimatedResidentBytes)}</dd>
                  <dt>address</dt>
                  <dd>{selectedAddressedNode.address}</dd>
                </dl>
                <section class="activity-section">
                  <h3>Pins</h3>
                  <div class="activity-group">
                    {#if selectedAddressedNode.pins.length === 0}
                      <small>none</small>
                    {:else}
                      {#each selectedAddressedNode.pins as pin}
                        <span class:consumer={pin.kind === 'display-view'} class:producer={pin.kind === 'display-producer'} class="pin-row">{pin.owner}</span>
                      {/each}
                    {/if}
                  </div>
                </section>
              {/if}
            {:else if selectedInspectorClient}
              <h3>{selectedInspectorClient.id}</h3>
              <dl>
                <dt>shape</dt>
                <dd>{selectedInspectorClient.shapeLabel}</dd>
                <dt>streams</dt>
                <dd>{selectedInspectorClient.subscriberCount}</dd>
                <dt>pinned segments</dt>
                <dd>{selectedInspectorClient.pinnedSegments.length}</dd>
                <dt>known pinned bytes</dt>
                <dd>{formatBytes(selectedInspectorClient.pinnedBytes)}</dd>
                <dt>last activity</dt>
                <dd>{formatTime(selectedInspectorClient.lastActivityAt)}</dd>
              </dl>
              <section class="activity-section">
                <h3>Consumer Pins</h3>
                <div class="activity-group">
                  {#each selectedInspectorClient.pinnedSegments as pin}
                    <button
                      class="pin-row consumer"
                      onclick={() => inspectorSelection = { kind: 'addressed', node: pin.node }}
                    >
                      {pin.node.path.join(' / ')}
                    </button>
                  {/each}
                </div>
              </section>
            {:else if selectedInspectorProjection}
              <h3>{selectedInspectorProjection.name}</h3>
              <dl>
                <dt>kind</dt>
                <dd>{selectedInspectorProjection.kind}</dd>
                <dt>summary</dt>
                <dd>{projectionSummary(selectedInspectorProjection)}</dd>
                <dt>fork</dt>
                <dd>{selectedInspectorProjection.forkId ? shortId(selectedInspectorProjection.forkId) : 'global'}</dd>
              </dl>
              <button
                class="secondary-action"
                onclick={() => {
                  selectedProjectionName = selectedInspectorProjection.name
                  activeTab = 'projections'
                }}
              >
                Inspect JSON
              </button>
            {:else if selectedIntrospection}
              <h3>{selectedIntrospection.session.title}</h3>
              <dl>
                <dt>session</dt>
                <dd>{selectedIntrospection.session.sessionId}</dd>
                <dt>cwd</dt>
                <dd>{selectedIntrospection.session.cwd}</dd>
                <dt>updated</dt>
                <dd>{formatTime(selectedIntrospection.session.updatedAt)}</dd>
                <dt>view streams</dt>
                <dd>{displayViewSubscriberCount}</dd>
                <dt>producer pins</dt>
                <dd>{memoryStats.producerPinnedEntries}</dd>
              </dl>
            {:else}
              <div class="empty">Select a live session</div>
            {/if}
          </aside>
        </div>
      {:else if activeTab === 'projections'}
        <div class="projection-inspection">
          <aside class="projection-browser">
            <div class="pane-head">
              <h2>Projections</h2>
              <span>{projections.length}</span>
            </div>
            {#if projections.length === 0}
              <div class="empty">Waiting for projection state</div>
            {:else}
              <div class="projection-list">
                {#each projections as projection}
                  <button
                    class:selected={selectedProjection?.name === projection.name}
                    onclick={() => selectedProjectionName = projection.name}
                  >
                    <strong>{projectionLabel(projection)}</strong>
                    <span>{projection.kind}</span>
                    <small>{projectionSummary(projection)}</small>
                  </button>
                {/each}
              </div>
            {/if}
          </aside>

          <section class="json-panel">
            <div class="map-head">
              <div>
                <h2>Projection State</h2>
                <span>{selectedProjection ? projectionSummary(selectedProjection) : 'none'}</span>
              </div>
              {#if selectedProjection}
                <em>{selectedProjection.kind}{selectedProjection.forkId ? ` · ${shortId(selectedProjection.forkId)}` : ''}</em>
              {/if}
            </div>
            {#if selectedProjection}
              <pre>{JSON.stringify(selectedProjection.state, null, 2)}</pre>
            {:else}
              <div class="empty">Select a projection</div>
            {/if}
          </section>
        </div>
      {:else}
        <div class="rpc-layout">
          <section class="activity-panel">
            <div class="pane-head">
              <h2>Commands</h2>
              <span>{rpcCommands.length}</span>
            </div>
            <div class="activity-group padded">
              {#if rpcTraceError}
                <div class="empty">{rpcTraceError}</div>
              {/if}
              {#each rpcCommands as trace}
                <a class:running={trace.isRunning} class:error={trace.errorCount > 0} class="call command" href={`http://127.0.0.1:27686/api/traces/${trace.traceId}`} target="_blank" rel="noreferrer">
                  <strong>{trace.rpcName}</strong>
                  <em>{durationLabel(trace)}</em>
                  <small>{trace.spanCount} spans{trace.errorCount ? ` · ${trace.errorCount} errors` : ''}</small>
                </a>
              {/each}
            </div>
          </section>
          <section class="activity-panel">
            <div class="pane-head">
              <h2>Streams</h2>
              <span>{rpcStreams.length}</span>
            </div>
            <div class="activity-group padded">
              {#each rpcStreams as trace}
                <a class:running={trace.isRunning} class:error={trace.errorCount > 0} class="call stream" href={`http://127.0.0.1:27686/api/traces/${trace.traceId}`} target="_blank" rel="noreferrer">
                  <strong>{trace.rpcName}</strong>
                  <em>{durationLabel(trace)}</em>
                  <small>{trace.spanCount} spans{trace.errorCount ? ` · ${trace.errorCount} errors` : ''}</small>
                </a>
              {/each}
            </div>
          </section>
        </div>
      {/if}
    </section>
  </section>
</main>
