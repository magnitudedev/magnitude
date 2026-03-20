/**
 * DebugPanel — Left-side panel showing debug information.
 *
 * Tabs:
 * - Events: Real-time event log
 * - Projections: All projection states
 * - Forks: Fork tree visualization
 * - Memory: Message history and context usage
 * - Turn: Current turn execution state
 */

import { memo, useState } from 'react'
import { TextAttributes } from '@opentui/core'
import { useTheme } from '../hooks/use-theme'
import { Button } from './button'
import { BOX_CHARS } from '../utils/ui-constants'
import { writeTextToClipboard } from '../utils/clipboard'
import { textOf } from '@magnitudedev/agent'
import type { DebugSnapshot, AppEvent } from '@magnitudedev/agent'
import type { LogEntry } from '@magnitudedev/logger'

type DebugTab = 'events' | 'logs' | 'projections' | 'forks' | 'memory' | 'turn' | 'artifacts'

interface DebugPanelProps {
  onToggle: () => void
  debugSnapshot: DebugSnapshot | null
  events: AppEvent[]
  logs: LogEntry[]
}

export const DebugPanel = memo(function DebugPanel({ onToggle, debugSnapshot, events, logs }: DebugPanelProps) {
  const theme = useTheme()
  const [activeTab, setActiveTab] = useState<DebugTab>('projections')

const tabs: Array<{ id: DebugTab; label: string }> = [
    { id: 'events', label: 'Events' },
    { id: 'logs', label: 'Logs' },
    { id: 'projections', label: 'Projections' },
    { id: 'forks', label: 'Forks' },
    { id: 'memory', label: 'Memory' },
    { id: 'turn', label: 'Turn' },
    { id: 'artifacts', label: 'Artifacts' },
  ]

  return (
    <box style={{ flexDirection: 'column', flexGrow: 1 }}>
      <box
        style={{
          flexDirection: 'column',
          flexGrow: 1,
          borderStyle: 'single',
          borderColor: theme.foreground,
          customBorderChars: BOX_CHARS,
        }}
      >
        {/* Title bar with hide button */}
        <box
          style={{
            flexDirection: 'row',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexShrink: 0,
            paddingLeft: 1,
            paddingRight: 1,
          }}
        >
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
            Debug Panel
          </text>
          <Button onClick={onToggle}>
            <box
              style={{
                borderStyle: 'single',
                borderColor: theme.foreground,
                customBorderChars: BOX_CHARS,
                paddingLeft: 1,
                paddingRight: 1,
              }}
            >
              <text style={{ fg: theme.foreground }}>Hide</text>
            </box>
          </Button>
        </box>

        {/* Tab bar */}
        <box
          style={{
            flexDirection: 'row',
            flexShrink: 0,
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
          }}
        >
          {tabs.map((tab, idx) => (
            <box key={tab.id} style={{ flexDirection: 'row' }}>
              {idx > 0 && <text style={{ fg: theme.foreground }}> │ </text>}
              <Button onClick={() => setActiveTab(tab.id)}>
                <text
                  style={{
                    fg: activeTab === tab.id ? theme.primary : theme.foreground,
                  }}
                  attributes={activeTab === tab.id ? TextAttributes.BOLD : undefined}
                >
                  {tab.label}
                </text>
              </Button>
            </box>
          ))}
        </box>

        {/* Content area */}
        <scrollbox
          stickyScroll
          stickyStart="top"
          scrollX={false}
          scrollbarOptions={{ visible: false }}
          verticalScrollbarOptions={{ visible: false }}
          style={{
            flexGrow: 1,
            rootOptions: {
              flexGrow: 1,
              backgroundColor: 'transparent',
            },
            wrapperOptions: {
              border: false,
              backgroundColor: 'transparent',
            },
            contentOptions: {
              paddingLeft: 1,
              paddingRight: 1,
              paddingTop: 1,
            },
          }}
        >
          <box style={{ flexDirection: 'column' }}>
            {activeTab === 'events' && <EventsTab events={events} />}
            {activeTab === 'logs' && <LogsTab logs={logs} />}
            {activeTab === 'projections' && <ProjectionsTab debugSnapshot={debugSnapshot} />}
            {activeTab === 'forks' && <ForksTab debugSnapshot={debugSnapshot} />}
            {activeTab === 'memory' && <MemoryTab debugSnapshot={debugSnapshot} />}
            {activeTab === 'turn' && <TurnTab debugSnapshot={debugSnapshot} />}
            {activeTab === 'artifacts' && <ArtifactsTab debugSnapshot={debugSnapshot} />}
          </box>
        </scrollbox>
      </box>
    </box>
  )
})

// =============================================================================
// Events Tab
// =============================================================================

const EVENT_TYPE_COLORS: Record<string, string> = {
  session_initialized: '#88c0d0',
  user_message: '#a3be8c',
  turn_started: '#ebcb8b',
  turn_completed: '#ebcb8b',
  message_chunk: '#d8dee9',
  thinking_chunk: '#b48ead',
  message_end: '#d8dee9',
  tool_event: '#81a1c1',
  interrupt: '#bf616a',
  fork_started: '#8fbcbb',
  fork_completed: '#8fbcbb',
  fork_removed: '#8fbcbb',
}

// Events too noisy to show by default
const FILTERED_EVENT_TYPES = new Set(['message_chunk', 'thinking_chunk'])

function EventsTab({ events }: { events: AppEvent[] }) {
  const theme = useTheme()
  const [showAll, setShowAll] = useState(false)

  const filtered = showAll ? events : events.filter(e => !FILTERED_EVENT_TYPES.has(e.type))
  // Show last 100
  const visible = filtered.slice(-100)

  return (
    <box style={{ flexDirection: 'column' }}>
      <box style={{ flexDirection: 'row', justifyContent: 'space-between', paddingBottom: 1 }}>
        <text style={{ fg: theme.secondary }}>
          Events ({filtered.length}{!showAll ? ` / ${events.length} total` : ''})
        </text>
        <box style={{ flexDirection: 'row' }}>
          <CopyButton text={JSON.stringify(filtered, null, 2)} />
          <text style={{ fg: theme.muted }}> </text>
          <Button onClick={() => setShowAll(prev => !prev)}>
            <text style={{ fg: theme.primary }}>
              {showAll ? '[Filter]' : '[Show All]'}
            </text>
          </Button>
        </box>
      </box>
      {visible.length === 0 && (
        <text style={{ fg: theme.muted }}>No events yet</text>
      )}
      {visible.map((event, idx) => (
        <EventRow key={idx} event={event} />
      ))}
    </box>
  )
}

function EventRow({ event }: { event: AppEvent }) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(false)
  const color = EVENT_TYPE_COLORS[event.type] ?? theme.foreground

  const summary = getEventSummary(event)

  return (
    <box style={{ flexDirection: 'column' }}>
      <Button onClick={() => setExpanded(prev => !prev)}>
        <box style={{ flexDirection: 'row' }}>
          <text style={{ fg: theme.muted }}>{expanded ? '▼ ' : '▶ '}</text>
          <text style={{ fg: color }} attributes={TextAttributes.BOLD}>{event.type}</text>
          {event.forkId && <text style={{ fg: theme.muted }}> [{event.forkId.slice(0, 6)}]</text>}
          {summary && <text style={{ fg: theme.secondary }}> {summary}</text>}
        </box>
      </Button>
      {expanded && (
        <box style={{ paddingLeft: 4, paddingBottom: 1 }}>
          <text style={{ fg: theme.foreground }}>
            {JSON.stringify(event, null, 2)}
          </text>
        </box>
      )}
    </box>
  )
}

function getEventSummary(event: AppEvent): string {
  switch (event.type) {
    case 'user_message': return truncate(textOf(event.content), 40)
    case 'turn_started': return `turn=${event.turnId.slice(0, 8)}`
    case 'turn_completed': return event.result.success ? 'success' : `error: ${event.result.error}`
    case 'tool_event': return `${event.toolKey} ${event.event._tag}`
    case 'agent_created': return event.name
    case 'message_end': return ''
    default: return ''
  }
}

function truncate(s: string, max: number): string {
  const oneLine = s.replace(/\n/g, ' ')
  return oneLine.length > max ? oneLine.slice(0, max) + '...' : oneLine
}

// =============================================================================
// Logs Tab
// =============================================================================

const LOG_LEVEL_COLORS: Record<string, string> = {
  DEBUG: '#4c566a',
  INFO: '#a3be8c',
  WARN: '#ebcb8b',
  ERROR: '#bf616a',
}

function LogsTab({ logs }: { logs: LogEntry[] }) {
  const theme = useTheme()
  const [filter, setFilter] = useState<string | null>(null)

  const filtered = filter ? logs.filter(l => l.level === filter) : logs

  return (
    <box style={{ flexDirection: 'column' }}>
      <box style={{ flexDirection: 'row', justifyContent: 'space-between', paddingBottom: 1 }}>
        <text style={{ fg: theme.secondary }}>
          Logs ({filtered.length}{filter ? ` / ${logs.length} total` : ''})
        </text>
        <box style={{ flexDirection: 'row' }}>
          <CopyButton text={formatLogsForCopy(filtered)} />
          <text style={{ fg: theme.muted }}> </text>
          {(['DEBUG', 'INFO', 'WARN', 'ERROR'] as const).map((level, idx) => (
            <box key={level} style={{ flexDirection: 'row' }}>
              {idx > 0 && <text style={{ fg: theme.muted }}> </text>}
              <Button onClick={() => setFilter(prev => prev === level ? null : level)}>
                <text style={{ fg: filter === level ? LOG_LEVEL_COLORS[level] : theme.muted }} attributes={filter === level ? TextAttributes.BOLD : undefined}>
                  [{level}]
                </text>
              </Button>
            </box>
          ))}
        </box>
      </box>
      {filtered.length === 0 && (
        <text style={{ fg: theme.muted }}>No logs yet</text>
      )}
      {filtered.map((entry, idx) => {
        const color = LOG_LEVEL_COLORS[entry.level] ?? theme.foreground
        const { level, timestamp, msg, ...rest } = entry
        const extra = Object.keys(rest).length > 0 ? ` ${JSON.stringify(rest)}` : ''
        return (
          <box key={idx} style={{ flexDirection: 'row' }}>
            <text style={{ fg: color }} attributes={TextAttributes.BOLD}>{level.padEnd(5)} </text>
            <text style={{ fg: theme.muted }}>{timestamp.slice(11, 23)} </text>
            <text style={{ fg: theme.foreground }}>{msg ?? ''}{extra}</text>
          </box>
        )
      })}
    </box>
  )
}

// =============================================================================
// Projections Tab (existing — kept as-is)
// =============================================================================

function ProjectionsTab({ debugSnapshot }: { debugSnapshot: DebugSnapshot | null }) {
  const theme = useTheme()
  const [expandedProjections, setExpandedProjections] = useState<Set<string>>(new Set())

  const toggleProjection = (name: string) => {
    setExpandedProjections(prev => {
      const next = new Set(prev)
      if (next.has(name)) {
        next.delete(name)
      } else {
        next.add(name)
      }
      return next
    })
  }

  if (!debugSnapshot) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.secondary }}>Waiting for debug data...</text>
      </box>
    )
  }

  return (
    <box style={{ flexDirection: 'column' }}>
      <text style={{ fg: theme.secondary, paddingBottom: 1 }}>
        Projection States ({debugSnapshot.projections.length})
      </text>
      {debugSnapshot.projections.map((projection, idx) => (
        <ProjectionSection
          key={projection.name}
          projection={projection}
          isExpanded={expandedProjections.has(projection.name)}
          onToggle={() => toggleProjection(projection.name)}
          isLast={idx === debugSnapshot.projections.length - 1}
        />
      ))}
    </box>
  )
}

interface ProjectionSectionProps {
  projection: { name: string; state: unknown; timestamp: number }
  isExpanded: boolean
  onToggle: () => void
  isLast: boolean
}

function ProjectionSection({ projection, isExpanded, onToggle, isLast }: ProjectionSectionProps) {
  const theme = useTheme()
  const [isCopied, setIsCopied] = useState(false)

  const handleCopy = async () => {
    try {
      const json = JSON.stringify(projection.state, null, 2)
      await writeTextToClipboard(json)
      setIsCopied(true)
      setTimeout(() => setIsCopied(false), 2000)
    } catch {
      // Error already logged by writeTextToClipboard
    }
  }

  return (
    <box style={{ flexDirection: 'column', paddingBottom: isLast ? 0 : 1 }}>
      <box
        style={{
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Button onClick={onToggle}>
          <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>
            {isExpanded ? '▼' : '▶'} {projection.name}
          </text>
        </Button>
        <Button onClick={handleCopy}>
          <text style={{ fg: theme.foreground }}>
            {isCopied ? '[Copied]' : '[Copy]'}
          </text>
        </Button>
      </box>
      {isExpanded && (
        <box style={{ flexDirection: 'column', paddingLeft: 2, paddingTop: 1 }}>
          <text style={{ fg: theme.foreground }}>
            {JSON.stringify(projection.state, null, 2)}
          </text>
        </box>
      )}
    </box>
  )
}

// =============================================================================
// Forks Tab
// =============================================================================

interface ForkInstance {
  readonly forkId: string
  readonly parentForkId: string | null
  readonly name: string
  readonly status: 'running' | 'completed'
  readonly context: string
  readonly result?: unknown
  readonly createdAt: number
  readonly completedAt?: number
}

function ForksTab({ debugSnapshot }: { debugSnapshot: DebugSnapshot | null }) {
  const theme = useTheme()

  const forkProjection = debugSnapshot?.projections.find(p => p.name === 'ForkProjection')
  const forkState = forkProjection?.state as { forks: Map<string, ForkInstance> } | undefined

  // Map might come through as a plain object after JSON serialization in the snapshot
  const forks: ForkInstance[] = []
  if (forkState?.forks) {
    if (forkState.forks instanceof Map) {
      for (const fork of forkState.forks.values()) {
        forks.push(fork)
      }
    } else {
      // Handle if serialized as object
      const obj = forkState.forks as unknown as Record<string, ForkInstance>
      for (const fork of Object.values(obj)) {
        forks.push(fork)
      }
    }
  }

  if (!debugSnapshot) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.secondary }}>Waiting for debug data...</text>
      </box>
    )
  }

  if (forks.length === 0) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.muted }}>No forks active</text>
      </box>
    )
  }

  const statusColor = (status: string) => {
    switch (status) {
      case 'running': return theme.success
      case 'completed': return theme.primary
      default: return theme.foreground
    }
  }

  return (
    <box style={{ flexDirection: 'column' }}>
      <text style={{ fg: theme.secondary, paddingBottom: 1 }}>
        Forks ({forks.length})
      </text>
      {forks.map((fork, idx) => (
        <box key={fork.forkId} style={{ flexDirection: 'column', paddingBottom: idx < forks.length - 1 ? 1 : 0 }}>
          <box style={{ flexDirection: 'row' }}>
            <text style={{ fg: statusColor(fork.status) }} attributes={TextAttributes.BOLD}>
              {fork.status === 'running' ? '●' : fork.status === 'completed' ? '✓' : '○'}{' '}
            </text>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>{fork.name}</text>
            <text style={{ fg: theme.muted }}> [{fork.forkId.slice(0, 8)}]</text>
          </box>
          <box style={{ paddingLeft: 4, flexDirection: 'column' }}>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.secondary }}>Status: </text>
              <text style={{ fg: statusColor(fork.status) }}>{fork.status}</text>
            </box>
            {fork.parentForkId && (
              <box style={{ flexDirection: 'row' }}>
                <text style={{ fg: theme.secondary }}>Parent: </text>
                <text style={{ fg: theme.foreground }}>{fork.parentForkId.slice(0, 8)}</text>
              </box>
            )}
            {fork.context && (
              <box style={{ flexDirection: 'row' }}>
                <text style={{ fg: theme.secondary }}>Context: </text>
                <text style={{ fg: theme.foreground }}>{truncate(fork.context, 60)}</text>
              </box>
            )}
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.secondary }}>Active: </text>
              <text style={{ fg: theme.foreground }}>{formatDuration(fork.createdAt, fork.completedAt ?? Date.now())}</text>
            </box>
            {fork.result !== undefined && (
              <box style={{ flexDirection: 'row' }}>
                <text style={{ fg: theme.secondary }}>Result: </text>
                <text style={{ fg: theme.foreground }}>{truncate(JSON.stringify(fork.result), 60)}</text>
              </box>
            )}
          </box>
        </box>
      ))}
    </box>
  )
}

// =============================================================================
// Artifacts Tab
// =============================================================================

interface ArtifactInfo {
  name: string
  content: string
  syncPath: string | null
}

function ArtifactsTab({ debugSnapshot }: { debugSnapshot: DebugSnapshot | null }) {
  const theme = useTheme()
  const [expandedArtifacts, setExpandedArtifacts] = useState<Set<string>>(new Set())

  const artifactProjection = debugSnapshot?.projections.find(p => p.name === 'ArtifactProjection')
  const artifactState = artifactProjection?.state as { artifacts: Map<string, ArtifactInfo> } | undefined

  // Map might come through as a plain object after JSON serialization in the snapshot
  const artifacts: ArtifactInfo[] = []
  if (artifactState?.artifacts) {
    if (artifactState.artifacts instanceof Map) {
      for (const artifact of artifactState.artifacts.values()) {
        artifacts.push(artifact)
      }
    } else {
      // Handle if serialized as object
      const obj = artifactState.artifacts as unknown as Record<string, ArtifactInfo>
      for (const artifact of Object.values(obj)) {
        artifacts.push(artifact)
      }
    }
  }

  const toggleArtifact = (name: string) => {
    setExpandedArtifacts(prev => {
      const next = new Set(prev)
      if (next.has(name)) {
        next.delete(name)
      } else {
        next.add(name)
      }
      return next
    })
  }

  if (!debugSnapshot) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.secondary }}>Waiting for debug data...</text>
      </box>
    )
  }

  if (artifacts.length === 0) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.muted }}>No artifacts</text>
      </box>
    )
  }

  return (
    <box style={{ flexDirection: 'column' }}>
      <text style={{ fg: theme.secondary, paddingBottom: 1 }}>
        Artifacts ({artifacts.length})
      </text>
      {artifacts.map((artifact, idx) => (
        <box key={artifact.name} style={{ flexDirection: 'column', paddingBottom: idx < artifacts.length - 1 ? 1 : 0 }}>
          <Button onClick={() => toggleArtifact(artifact.name)}>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.muted }}>{expandedArtifacts.has(artifact.name) ? '▼ ' : '▶ '}</text>
              <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>
                ● {artifact.name}
              </text>
            </box>
          </Button>
          <box style={{ paddingLeft: 4, flexDirection: 'column' }}>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.secondary }}>Content: </text>
              <text style={{ fg: theme.foreground }}>{artifact.content.length} chars</text>
            </box>
            {artifact.syncPath && (
              <box style={{ flexDirection: 'row' }}>
                <text style={{ fg: theme.secondary }}>Sync path: </text>
                <text style={{ fg: theme.foreground }}>{artifact.syncPath}</text>
              </box>
            )}
            {!expandedArtifacts.has(artifact.name) && (
              <box style={{ flexDirection: 'row' }}>
                <text style={{ fg: theme.muted }}>{truncate(artifact.content, 200)}</text>
              </box>
            )}
            {expandedArtifacts.has(artifact.name) && (
              <box style={{ flexDirection: 'column', paddingTop: 1 }}>
                <text style={{ fg: theme.foreground }}>
                  {artifact.content}
                </text>
              </box>
            )}
          </box>
        </box>
      ))}
    </box>
  )
}

// =============================================================================
// Memory Tab (existing — kept as-is)
// =============================================================================

function MemoryTab({ debugSnapshot }: { debugSnapshot: DebugSnapshot | null }) {
  const theme = useTheme()

  if (!debugSnapshot?.contextUsage) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.secondary }}>No memory data available</text>
      </box>
    )
  }

  const { contextUsage } = debugSnapshot
  const memoryProjection = debugSnapshot.projections.find(p => p.name === 'MemoryProjection')
  const memoryState = memoryProjection?.state as { messages?: { role: string; content: string }[]; queuedMessages?: unknown[] } | undefined

  const barWidth = 50
  const filledWidth = Math.min(barWidth, Math.round((contextUsage.usagePercent / 100) * barWidth))
  const emptyWidth = barWidth - filledWidth

  let usageColor = theme.success
  if (contextUsage.usagePercent > 90) {
    usageColor = theme.error
  } else if (contextUsage.usagePercent > 70) {
    usageColor = theme.warning
  }

  return (
    <box style={{ flexDirection: 'column' }}>
      <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
        Context Usage
      </text>

      <box style={{ flexDirection: 'row', paddingTop: 1 }}>
        <text style={{ fg: theme.secondary }}>Tokens: </text>
        <text style={{ fg: theme.foreground }}>
          {contextUsage.currentTokens.toLocaleString()} / {contextUsage.hardCap.toLocaleString()} (compact at {contextUsage.softCap.toLocaleString()})
        </text>
      </box>

      <box style={{ flexDirection: 'row' }}>
        <text style={{ fg: theme.secondary }}>Messages: </text>
        <text style={{ fg: theme.foreground }}>{contextUsage.messageCount}</text>
      </box>

      <box style={{ flexDirection: 'row' }}>
        <text style={{ fg: theme.secondary }}>Usage: </text>
        <text style={{ fg: usageColor }} attributes={TextAttributes.BOLD}>
          {contextUsage.usagePercent}%
        </text>
      </box>

      <box style={{ flexDirection: 'row', paddingTop: 1 }}>
        <text style={{ fg: usageColor }}>{'█'.repeat(filledWidth)}</text>
        <text style={{ fg: theme.secondary }}>{'░'.repeat(emptyWidth)}</text>
      </box>

      {contextUsage.shouldCompact && (
        <box style={{ flexDirection: 'row', paddingTop: 1 }}>
          <text style={{ fg: theme.warning }}>
            {contextUsage.isCompacting ? 'Compacting...' : 'Should compact'}
          </text>
        </box>
      )}

      {memoryState?.messages && (
        <>
          <box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingTop: 2 }}>
            <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
              Message History ({memoryState.messages.length})
            </text>
            <CopyButton text={memoryState.messages.map((m, i) => `[${i}] ${m.role}:\n${m.content}`).join('\n\n')} />
          </box>

          <box style={{ flexDirection: 'column', paddingTop: 1 }}>
            {memoryState.messages.map((msg, idx) => (
              <MessageItem key={idx} msg={msg} idx={idx} />
            ))}
          </box>
        </>
      )}

      {memoryState?.queuedMessages && memoryState.queuedMessages.length > 0 && (
        <>
          <text style={{ fg: theme.warning, paddingTop: 2 }} attributes={TextAttributes.BOLD}>
            Queued Messages ({memoryState.queuedMessages.length})
          </text>
          <text style={{ fg: theme.secondary }}>
            Waiting to be flushed on next turn
          </text>
        </>
      )}
    </box>
  )
}

// =============================================================================
// Turn Tab
// =============================================================================

interface ToolCallInfo {
  readonly toolCallId: string
  readonly toolKey: string
  readonly input: unknown
  readonly result?: { readonly status: string; readonly output?: unknown; readonly message?: string }
}

interface TurnStateInfo {
  readonly activeTurn: {
    readonly turnId: string
    readonly chainId: string
    readonly toolCalls: readonly ToolCallInfo[]
  } | null
  readonly completedTurns: number
}

function TurnTab({ debugSnapshot }: { debugSnapshot: DebugSnapshot | null }) {
  const theme = useTheme()
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set())

  const turnProjection = debugSnapshot?.projections.find(p => p.name === 'TurnProjection')
  const turnState = turnProjection?.state as TurnStateInfo | undefined

  if (!debugSnapshot) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.secondary }}>Waiting for debug data...</text>
      </box>
    )
  }

  if (!turnState) {
    return (
      <box style={{ flexDirection: 'column' }}>
        <text style={{ fg: theme.muted }}>No turn data</text>
      </box>
    )
  }

  const toggleTool = (id: string) => {
    setExpandedTools(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const resultColor = (status: string) => {
    switch (status) {
      case 'success': return theme.success
      case 'error': return theme.error
      case 'rejected': return theme.warning
      case 'interrupted': return theme.muted
      default: return theme.foreground
    }
  }

  return (
    <box style={{ flexDirection: 'column' }}>
      <box style={{ flexDirection: 'row' }}>
        <text style={{ fg: theme.secondary }}>Completed turns: </text>
        <text style={{ fg: theme.foreground }}>{turnState.completedTurns}</text>
      </box>

      {!turnState.activeTurn ? (
        <text style={{ fg: theme.muted, paddingTop: 1 }}>No active turn (idle)</text>
      ) : (
        <box style={{ flexDirection: 'column', paddingTop: 1 }}>
          <text style={{ fg: theme.foreground }} attributes={TextAttributes.BOLD}>
            Active Turn
          </text>
          <box style={{ paddingLeft: 2, flexDirection: 'column' }}>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.secondary }}>Turn ID: </text>
              <text style={{ fg: theme.foreground }}>{turnState.activeTurn.turnId.slice(0, 12)}</text>
            </box>
            <box style={{ flexDirection: 'row' }}>
              <text style={{ fg: theme.secondary }}>Chain ID: </text>
              <text style={{ fg: theme.foreground }}>{turnState.activeTurn.chainId.slice(0, 12)}</text>
            </box>

            <text style={{ fg: theme.foreground, paddingTop: 1 }} attributes={TextAttributes.BOLD}>
              Tool Calls ({turnState.activeTurn.toolCalls.length})
            </text>

            {turnState.activeTurn.toolCalls.map((tc) => {
              const isExpanded = expandedTools.has(tc.toolCallId)
              return (
                <box key={tc.toolCallId} style={{ flexDirection: 'column', paddingTop: 1 }}>
                  <Button onClick={() => toggleTool(tc.toolCallId)}>
                    <box style={{ flexDirection: 'row' }}>
                      <text style={{ fg: theme.muted }}>{isExpanded ? '▼ ' : '▶ '}</text>
                      <text style={{ fg: theme.primary }} attributes={TextAttributes.BOLD}>
                        {tc.toolKey}
                      </text>
                      {tc.result && (
                        <text style={{ fg: resultColor(tc.result.status) }}>
                          {' '}[{tc.result.status}]
                        </text>
                      )}
                      {!tc.result && (
                        <text style={{ fg: theme.warning }}> [running]</text>
                      )}
                    </box>
                  </Button>
                  {isExpanded && (
                    <box style={{ paddingLeft: 4, flexDirection: 'column' }}>
                      <text style={{ fg: theme.secondary }}>Input:</text>
                      <text style={{ fg: theme.foreground }}>
                        {truncate(JSON.stringify(tc.input), 200)}
                      </text>
                      {tc.result && (
                        <>
                          <text style={{ fg: theme.secondary, paddingTop: 1 }}>Result ({tc.result.status}):</text>
                          <text style={{ fg: theme.foreground }}>
                            {tc.result.status === 'success'
                              ? truncate(JSON.stringify(tc.result.output), 200)
                              : tc.result.message ?? tc.result.status}
                          </text>
                        </>
                      )}
                    </box>
                  )}
                </box>
              )
            })}
          </box>
        </box>
      )}
    </box>
  )
}

// =============================================================================
// Helpers
// =============================================================================

function MessageItem({ msg, idx }: { msg: { role: string; content: string }; idx: number }) {
  const theme = useTheme()
  const [expanded, setExpanded] = useState(true)

  return (
    <box
      style={{
        flexDirection: 'column',
        paddingLeft: 1,
        paddingRight: 1,
        backgroundColor: msg.role === 'user' ? '#1a2a1a' : '#1a1a2e',
      }}
    >
      <box style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button onClick={() => setExpanded(prev => !prev)}>
          <text
            style={{ fg: msg.role === 'user' ? theme.success : theme.primary }}
            attributes={TextAttributes.BOLD}
          >
            {expanded ? '▼' : '▶'} [{idx}] {msg.role === 'user' ? 'User' : 'Assistant'}
          </text>
        </Button>
        <CopyButton text={msg.content} />
      </box>
      {expanded && (
        <text style={{ fg: theme.foreground, paddingLeft: 2, paddingBottom: 1 }}>
          {msg.content}
        </text>
      )}
    </box>
  )
}

function CopyButton({ text }: { text: string }) {
  const theme = useTheme()
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await writeTextToClipboard(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Error already logged by writeTextToClipboard
    }
  }

  return (
    <Button onClick={handleCopy}>
      <text style={{ fg: copied ? theme.success : theme.muted }}>
        {copied ? '[Copied]' : '[Copy]'}
      </text>
    </Button>
  )
}

function formatLogsForCopy(logs: LogEntry[]): string {
  return logs.map(entry => {
    const { level, timestamp, msg, ...rest } = entry
    const extra = Object.keys(rest).length > 0 ? ` ${JSON.stringify(rest)}` : ''
    return `${level.padEnd(5)} ${timestamp} ${msg ?? ''}${extra}`
  }).join('\n')
}

function formatDuration(startMs: number, endMs: number): string {
  const seconds = Math.floor((endMs - startMs) / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}m ${remainingSeconds}s`
}
