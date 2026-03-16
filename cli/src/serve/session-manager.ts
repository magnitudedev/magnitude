import { Layer } from 'effect'
import type { StorageClient } from '@magnitudedev/storage'
import { resolve } from 'path'
import { stat } from 'fs/promises'
import {
  createCodingAgentClient,
  collectSessionContext,
  ChatPersistence,
  textParts,
  type AppEvent,
  type DisplayState,
  type AgentStatusState,
  type ForkWorkingState,
  type ArtifactState,
} from '@magnitudedev/agent'
import { JsonChatPersistence } from '../persistence'

type AgentClient = Awaited<ReturnType<typeof createCodingAgentClient>>

export interface SessionInfo {
  id: string
  title: string
  status: 'idle' | 'streaming'
  createdAt: string
  cwd: string
}

export interface SessionDetail extends SessionInfo {
  display: DisplayState
  forkState: AgentStatusState
  workingState: ForkWorkingState
  artifactState: ArtifactState
}

interface SessionRecord {
  id: string
  createdAt: string
  title: string
  cwd: string
  client: AgentClient
  status: 'idle' | 'streaming'
  display: DisplayState | null
  forkState: AgentStatusState | null
  workingState: ForkWorkingState | null
  artifactState: ArtifactState | null
  eventSeq: number
  eventBuffer: SseEnvelope[]
  subscribers: Set<(evt: SseEnvelope) => void>
  unsubscribers: Array<() => void>
  lastDisplayEmitAt: number
  displayTimer: ReturnType<typeof setTimeout> | null
}

interface SseEnvelope {
  id: string
  event: 'agent_event' | 'display_state' | 'fork_state' | 'working_state' | 'artifact_state'
  data: unknown
}

interface GlobalSseEnvelope extends SseEnvelope {
  sessionId: string
}


const DEFAULT_DISPLAY_STATE: DisplayState = {
  status: 'idle',
  messages: [],
  currentTurnId: null,
  streamingMessageId: null,
  activeThinkBlockId: null,
  showButton: 'send',
  colorAssignments: new Map(),
}

const DEFAULT_FORK_STATE: AgentStatusState = {
  agents: new Map(),
  agentByForkId: new Map(),
}

const DEFAULT_WORKING_STATE: ForkWorkingState = {
  working: false,
  willContinue: false,
  hasQueuedMessages: false,
  pendingWake: false,
  currentChainId: null,
  currentTurnId: null,
  compactionPending: false,
  contextLimitBlocked: false,
  pendingApproval: false,
  softInterrupted: false,
  pendingMentionTimestamps: [],
}

const DEFAULT_ARTIFACT_STATE: ArtifactState = {
  artifacts: new Map(),
}

export class SessionNotFoundError extends Error {
  constructor(id: string) {
    super(`Session not found: ${id}`)
    this.name = 'SessionNotFoundError'
  }
}

export interface CreateSessionOptions {
  cwd?: string
}

export class SessionManager {
  private readonly debug: boolean
  private readonly sessions = new Map<string, SessionRecord>()
  private globalSeq = 0
  private globalBuffer: GlobalSseEnvelope[] = []
  private globalSubscribers = new Set<(evt: GlobalSseEnvelope) => void>()

  private readonly storage: StorageClient

  constructor(opts: { debug: boolean; storage: StorageClient }) {
    this.debug = opts.debug
    this.storage = opts.storage
  }

  async createSession(opts?: CreateSessionOptions): Promise<SessionInfo> {
    const id = this.storage.sessions.createId()
    const requestedCwd = opts?.cwd ? resolve(opts.cwd) : process.cwd()
    const cwdStat = await stat(requestedCwd).catch(() => null)
    if (!cwdStat?.isDirectory()) {
      throw new Error(`Invalid cwd: ${requestedCwd}`)
    }

    const persistence = new JsonChatPersistence({
      storage: this.storage,
      workingDirectory: requestedCwd,
      sessionId: id,
    })
    const layer = Layer.succeed(ChatPersistence, persistence)

    const sessionContext = await collectSessionContext({ cwd: requestedCwd, storage: this.storage })
    const client = await createCodingAgentClient({ persistence: layer, storage: this.storage, debug: this.debug, sessionContext })
    const createdAt = new Date().toISOString()

    const record: SessionRecord = {
      id,
      createdAt,
      title: 'New Chat',
      cwd: sessionContext.cwd,
      client,
      status: 'idle',
      display: null,
      forkState: null,
      workingState: null,
      artifactState: null,
      eventSeq: 0,
      eventBuffer: [],
      subscribers: new Set(),
      unsubscribers: [],
      lastDisplayEmitAt: 0,
      displayTimer: null,
    }

    record.unsubscribers.push(client.onEvent((event: AppEvent) => {
      this.pushSessionEvent(record, 'agent_event', event)
    }))

    record.unsubscribers.push(client.state.display.subscribeFork(null, (state) => {
      record.display = state
      record.status = state.status
      this.emitThrottledDisplay(record)
    }))

    record.unsubscribers.push(client.state.agentStatus.subscribe((state) => {
      record.forkState = state
      this.pushSessionEvent(record, 'fork_state', state)
    }))

    record.unsubscribers.push(client.state.working.subscribeFork(null, (state) => {
      record.workingState = state
      this.pushSessionEvent(record, 'working_state', state)
    }))

    record.unsubscribers.push(client.state.artifacts.subscribe((state) => {
      record.artifactState = state
      this.pushSessionEvent(record, 'artifact_state', state)
    }))

    record.unsubscribers.push(client.on.chatTitleGenerated(({ title }) => {
      record.title = title
    }))

    this.sessions.set(id, record)

    return {
      id,
      title: record.title,
      status: record.status,
      createdAt: record.createdAt,
      cwd: record.cwd,
    }
  }

  listSessions(): SessionInfo[] {
    return [...this.sessions.values()].map((record) => ({
      id: record.id,
      title: record.title,
      status: record.status,
      createdAt: record.createdAt,
      cwd: record.cwd,
    }))
  }

  getSessionInfo(id: string): SessionInfo | null {
    const record = this.sessions.get(id)
    if (!record) return null
    return {
      id: record.id,
      title: record.title,
      status: record.status,
      createdAt: record.createdAt,
      cwd: record.cwd,
    }
  }

  getSessionDetail(id: string): SessionDetail | null {
    const record = this.sessions.get(id)
    if (!record) return null
    return {
      id: record.id,
      title: record.title,
      status: record.status,
      createdAt: record.createdAt,
      cwd: record.cwd,
      display: record.display ?? DEFAULT_DISPLAY_STATE,
      forkState: record.forkState ?? DEFAULT_FORK_STATE,
      workingState: record.workingState ?? DEFAULT_WORKING_STATE,
      artifactState: record.artifactState ?? DEFAULT_ARTIFACT_STATE,
    }
  }

  async deleteSession(id: string): Promise<boolean> {
    const record = this.sessions.get(id)
    if (!record) return false

    if (record.displayTimer) {
      clearTimeout(record.displayTimer)
      record.displayTimer = null
    }

    for (const unsub of record.unsubscribers) {
      unsub()
    }

    await record.client.dispose()
    this.sessions.delete(id)
    return true
  }

  async disposeAll(): Promise<void> {
    const ids = [...this.sessions.keys()]
    for (const id of ids) {
      await this.deleteSession(id)
    }
  }

  async sendUserMessage(id: string, content: string): Promise<void> {
    const record = this.requireSession(id)
    if (!content.trim()) {
      throw new Error('Message content cannot be empty')
    }
    await record.client.send({
      type: 'user_message',
      forkId: null,
      content: textParts(content),
      attachments: [],
      mode: 'text',
      synthetic: false,
      taskMode: false,
    })
  }

  async interrupt(id: string): Promise<void> {
    const record = this.requireSession(id)
    await record.client.send({ type: 'interrupt', forkId: null })
  }

  async approveTool(id: string, toolCallId: string): Promise<void> {
    const record = this.requireSession(id)
    await record.client.send({ type: 'tool_approved', forkId: null, toolCallId })
  }

  async rejectTool(id: string, toolCallId: string, reason?: string): Promise<void> {
    const record = this.requireSession(id)
    await record.client.send({ type: 'tool_rejected', forkId: null, toolCallId, reason })
  }

  subscribeSessionEvents(
    id: string,
    cb: (evt: SseEnvelope) => void,
    lastEventId?: string | null
  ): { replay: SseEnvelope[]; unsubscribe: () => void } {
    const record = this.requireSession(id)
    const replay = this.replayFromBuffer(record.eventBuffer, lastEventId)
    record.subscribers.add(cb)
    return {
      replay,
      unsubscribe: () => {
        record.subscribers.delete(cb)
      },
    }
  }

  subscribeGlobalEvents(
    cb: (evt: GlobalSseEnvelope) => void,
    lastEventId?: string | null
  ): { replay: GlobalSseEnvelope[]; unsubscribe: () => void } {
    const replay = this.replayFromBuffer(this.globalBuffer, lastEventId)
    this.globalSubscribers.add(cb)
    return {
      replay,
      unsubscribe: () => {
        this.globalSubscribers.delete(cb)
      },
    }
  }

  private requireSession(id: string): SessionRecord {
    const record = this.sessions.get(id)
    if (!record) throw new SessionNotFoundError(id)
    return record
  }

  private emitThrottledDisplay(record: SessionRecord): void {
    const now = Date.now()
    const elapsed = now - record.lastDisplayEmitAt

    if (elapsed >= 100) {
      record.lastDisplayEmitAt = now
      this.pushSessionEvent(record, 'display_state', record.display ?? DEFAULT_DISPLAY_STATE)
      return
    }

    if (record.displayTimer) return

    const waitMs = 100 - elapsed
    record.displayTimer = setTimeout(() => {
      record.displayTimer = null
      record.lastDisplayEmitAt = Date.now()
      this.pushSessionEvent(record, 'display_state', record.display ?? DEFAULT_DISPLAY_STATE)
    }, waitMs)
  }

  private pushSessionEvent(record: SessionRecord, event: SseEnvelope['event'], data: unknown): void {
    const envelope: SseEnvelope = {
      id: String(++record.eventSeq),
      event,
      data,
    }

    record.eventBuffer.push(envelope)
    if (record.eventBuffer.length > 1000) {
      record.eventBuffer.shift()
    }

    for (const sub of record.subscribers) {
      sub(envelope)
    }

    const globalEnvelope: GlobalSseEnvelope = {
      sessionId: record.id,
      id: String(++this.globalSeq),
      event,
      data,
    }

    this.globalBuffer.push(globalEnvelope)
    if (this.globalBuffer.length > 1000) {
      this.globalBuffer.shift()
    }

    for (const sub of this.globalSubscribers) {
      sub(globalEnvelope)
    }
  }

  private replayFromBuffer<T extends { id: string }>(buffer: T[], lastEventId?: string | null): T[] {
    if (lastEventId == null) return [...buffer]
    const last = Number(lastEventId)
    if (!Number.isFinite(last)) return [...buffer]
    return buffer.filter((evt) => Number(evt.id) > last)
  }
}