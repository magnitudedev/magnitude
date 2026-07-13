import ansis from 'ansis'

type ToolEvent = Extract<AppEvent, { type: 'tool_event' }>
type ToolLifecycle = ToolEvent['event']
type ToolResult = Extract<ToolLifecycle, { _tag: 'ToolExecutionEnded' }>['result']
type ToolResultTag = 'Success' | 'Error' | 'Denied' | 'Interrupted' | 'InputRejected'
type ToolResultLike = { readonly _tag: ToolResultTag; readonly [key: string]: unknown }

type UserPart = { readonly _tag: 'TextPart'; readonly text?: string } | { readonly _tag: string; readonly text?: string }
type TurnOutcome = { readonly _tag: string; readonly [key: string]: any }
type ToolLifecycleEvent =
  | { readonly _tag: 'ToolInputFieldChunk'; readonly field: string; readonly delta: string }
  | { readonly _tag: 'ToolInputFieldComplete'; readonly field: string; readonly value: unknown }
  | { readonly _tag: 'ToolExecutionStarted'; readonly input: Record<string, unknown> }
  | { readonly _tag: 'ToolExecutionEnded'; readonly result: ToolResultLike }

export type AppEvent =
  | { readonly type: 'user_message'; readonly content: readonly UserPart[]; readonly synthetic: boolean; readonly taskMode: boolean }
  | { readonly type: 'message_start'; readonly forkId: string | null; readonly turnId: string; readonly id: string; readonly destination: { readonly kind: string } }
  | { readonly type: 'message_chunk'; readonly forkId: string | null; readonly turnId: string; readonly id: string; readonly text: string }
  | { readonly type: 'message_end'; readonly forkId: string | null; readonly turnId: string; readonly id: string }
  | { readonly type: 'agent_created'; readonly forkId: string; readonly agentId: string; readonly role: string; readonly name: string }
  | { readonly type: 'agent_killed'; readonly forkId: string; readonly reason: string }
  | { readonly type: 'worker_user_killed'; readonly forkId: string }
  | { readonly type: 'worker_idle_closed'; readonly forkId: string }
  | { readonly type: 'tool_event'; readonly forkId: string | null; readonly toolCallId: string; readonly toolKey: string; readonly event: ToolLifecycleEvent }
  | { readonly type: 'turn_outcome'; readonly forkId: string | null; readonly outcome: TurnOutcome }
  | { readonly type: 'interrupt'; readonly forkId: string | null }

export interface HeadlessOutput {
  readonly lines: readonly string[]
  readonly toolCount: number
}

interface AgentInfo {
  readonly forkId: string
  readonly agentId: string
  readonly role: string
  readonly name: string
}

interface BufferedMessage {
  readonly forkId: string | null
  readonly turnId: string
  readonly destination: Extract<AppEvent, { type: 'message_start' }>['destination']
  text: string
}

interface ToolRecord {
  readonly toolCallId: string
  readonly forkId: string | null
  readonly toolKey: string
  readonly input: Record<string, unknown>
  result?: ToolResult
}

const hiddenTools = new Set([
  'createTask',
  'updateTask',
  'killWorker',
  'reassignWorker',
  'messageWorker',
  'messageAdvisor',
  'finishGoal',
  'compact',
])

const ok = ansis.hex('#1f9670').bold
const err = ansis.hex('#f87171').bold
const dim = ansis.hex('#94a3b8')

export function createHeadlessOutputRenderer() {
  const agentsByFork = new Map<string, AgentInfo>()
  const agentsById = new Map<string, AgentInfo>()
  const announcedAgents = new Set<string>()
  const messages = new Map<string, BufferedMessage>()
  const tools = new Map<string, ToolRecord>()
  const completedTools = new Set<string>()
  let toolCount = 0

  function handleEvent(event: AppEvent): HeadlessOutput {
    const lines: string[] = []

    switch (event.type) {
      case 'user_message': {
        const text = textFromParts(event.content)
        if (text.trim()) {
          lines.push(event.synthetic ? `> [autopilot] ${text}` : renderUserMessage(text, event.taskMode))
        }
        break
      }

      case 'message_start': {
        if (event.destination.kind !== 'user') break
        messages.set(event.id, {
          forkId: event.forkId,
          turnId: event.turnId,
          destination: event.destination,
          text: '',
        })
        break
      }

      case 'message_chunk': {
        const msg = messages.get(event.id)
        if (!msg || msg.turnId !== event.turnId || msg.forkId !== event.forkId) break
        msg.text += event.text
        break
      }

      case 'message_end': {
        const msg = messages.get(event.id)
        if (!msg || msg.turnId !== event.turnId || msg.forkId !== event.forkId) break
        messages.delete(event.id)

        // Headless is a coherent log, not a live stream. Only completed root
        // user-facing messages are emitted as prose; worker messages are covered
        // by worker/tool progress lines.
        if (msg.forkId === null) {
          const text = msg.text.trim()
          if (text) lines.push(text)
        }
        break
      }

      case 'agent_created': {
        const agent: AgentInfo = {
          forkId: event.forkId,
          agentId: event.agentId,
          role: event.role,
          name: event.name,
        }
        agentsByFork.set(event.forkId, agent)
        agentsById.set(event.agentId, agent)
        if (!announcedAgents.has(event.agentId)) {
          announcedAgents.add(event.agentId)
          lines.push(renderAgentStart(agent))
        }
        break
      }

      case 'agent_killed': {
        const agent = agentForFork(event.forkId, agentsByFork)
        lines.push(`■ ${agentLabel(agent, event.forkId)} killed · ${event.reason}`)
        break
      }

      case 'worker_user_killed': {
        const agent = agentForFork(event.forkId, agentsByFork)
        lines.push(`■ ${agentLabel(agent, event.forkId)} stopped by user`)
        break
      }

      case 'worker_idle_closed': {
        const agent = agentForFork(event.forkId, agentsByFork)
        lines.push(`✓ ${agentLabel(agent, event.forkId)} closed`)
        break
      }

      case 'tool_event': {
        const output = handleToolEvent(event, tools, completedTools, agentsByFork, agentsById, announcedAgents)
        if (output) {
          toolCount++
          lines.push(output)
        }
        break
      }

      case 'turn_outcome': {
        if (event.forkId !== null) {
          const agent = agentForFork(event.forkId, agentsByFork)
          if (!outcomeWillContinue(event.outcome)) {
            lines.push(renderWorkerOutcome(agent, event.outcome, event.forkId))
          }
        } else {
          const line = renderRootOutcome(event.outcome)
          if (line) lines.push(line)
        }
        break
      }

      case 'interrupt':
        lines.push(dim(event.forkId === null ? '■ Interrupted' : '■ Worker interrupted'))
        break
    }

    return { lines, toolCount }
  }

  function getToolCount() {
    return toolCount
  }

  return { handleEvent, getToolCount }
}

export function renderUsageSummary(elapsedMs: number, totalTools: number, success: boolean): string {
  const label = success ? 'Finished' : 'Failed'
  const icon = success ? '✓' : '✗'
  const color = success ? ok : err
  return color(`${icon} ${label} · ${formatDuration(Math.floor(elapsedMs / 1000))} · ${totalTools} ${totalTools === 1 ? 'tool' : 'tools'}`)
}

export function renderErrorMessage(message: string): string {
  return err(`✗ ${message}`)
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60)
  const s = seconds % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

function handleToolEvent(
  event: ToolEvent,
  tools: Map<string, ToolRecord>,
  completedTools: Set<string>,
  agentsByFork: Map<string, AgentInfo>,
  agentsById: Map<string, AgentInfo>,
  announcedAgents: Set<string>,
): string | null {
  const inner = event.event
  const key = event.toolCallId
  let record = tools.get(key)

  if (!record) {
    record = {
      toolCallId: key,
      forkId: event.forkId,
      toolKey: event.toolKey,
      input: {},
    }
    tools.set(key, record)
  }

  if (inner._tag === 'ToolInputFieldChunk') {
    const current = typeof record.input[inner.field] === 'string' ? String(record.input[inner.field]) : ''
    record.input[inner.field] = current + inner.delta
    return null
  }

  if (inner._tag === 'ToolInputFieldComplete') {
    record.input[inner.field] = inner.value
    return null
  }

  if (inner._tag === 'ToolExecutionStarted') {
    Object.assign(record.input, inner.input)
    return null
  }

  if (inner._tag !== 'ToolExecutionEnded') return null
  if (completedTools.has(key)) return null
  completedTools.add(key)
  record.result = inner.result

  if (hiddenTools.has(event.toolKey)) return null

  if (event.toolKey === 'spawnWorker') {
    const agentId = stringValue(record.input.agentId) ?? stringValue(outputObject(inner.result)?.agentId)
    if (agentId && announcedAgents.has(agentId)) return null
  }

  const line = renderTool(record)

  if (event.toolKey === 'spawnWorker') {
    const agentId = stringValue(record.input.agentId) ?? stringValue(outputObject(inner.result)?.agentId)
    if (agentId) {
      announcedAgents.add(agentId)
      const role = stringValue(record.input.role)
      const title = stringValue(outputObject(inner.result)?.title) ?? stringValue(record.input.taskId) ?? stringValue(record.input.message)
      agentsById.set(agentId, {
        forkId: agentId,
        agentId,
        role: role ?? 'agent',
        name: String(title ?? agentId),
      })
    }
  }

  return prefixed(record.forkId, line, agentsByFork)
}

function renderTool(record: ToolRecord): string {
  const { toolKey, input, result } = record
  const status = resultTag(result)
  const output = successOutput(result)

  if (status === 'Error') return `✗ ${toolKey} · ${toolErrorMessage(result)}`
  if (status === 'Denied') return `■ ${toolKey} denied · ${stringify(toolField(result, 'denial'))}`
  if (status === 'Interrupted') return `■ ${toolKey} interrupted`
  if (status === 'InputRejected') return `✗ ${toolKey} rejected · ${toolIssueMessage(result)}`

  switch (toolKey) {
    case 'fileRead': {
      const path = stringValue(input.path)
      const content = typeof output === 'string' ? output : ''
      const lines = content ? content.split('\n').length : null
      return `→ Read ${path ?? '(unknown)'}${lines != null ? ` · ${lines} ${lines === 1 ? 'line' : 'lines'}` : ''}`
    }
    case 'fileWrite': {
      const path = stringValue(input.path)
      const content = stringValue(input.content)
      const lines = content ? content.split('\n').length : null
      return `→ Write ${path ?? '(unknown)'}${lines != null ? ` · ${lines} ${lines === 1 ? 'line' : 'lines'}` : ''}`
    }
    case 'fileEdit': {
      const path = stringValue(input.path)
      const detailText = stringValue(output)
      const detail = detailText ? ` · ${detailText}` : ''
      return `✎ Edit ${path ?? '(unknown)'}${detail}`
    }
    case 'fileSearch': {
      const pattern = stringValue(input.pattern) ?? ''
      const path = stringValue(input.path)
      const glob = stringValue(input.glob)
      const matches = Array.isArray(output) ? output : []
      const files = new Set(matches
        .map((match) => objectValue(match)?.file)
        .filter((file): file is string => typeof file === 'string' && file.length > 0)).size
      const scope = [path, glob].filter(Boolean).join(' ')
      return `/ Search "${pattern}"${scope ? ` in ${scope}` : ''} · ${matches.length} ${matches.length === 1 ? 'match' : 'matches'} in ${files} ${files === 1 ? 'file' : 'files'}`
    }
    case 'fileTree': {
      const path = stringValue(input.path) ?? '.'
      const entries = Array.isArray(output) ? output : []
      const files = entries.filter((entry) => objectValue(entry)?.type === 'file').length
      const dirs = entries.filter((entry) => objectValue(entry)?.type === 'dir').length
      return `◫ List ${path} · ${files} ${files === 1 ? 'file' : 'files'}${dirs ? `, ${dirs} ${dirs === 1 ? 'dir' : 'dirs'}` : ''}`
    }
    case 'shell': {
      const command = stringValue(input.command) ?? '(unknown command)'
      const output = outputObject(result)
      const exitCode = typeof output?.exitCode === 'number' ? output.exitCode : null
      return `$ ${command}${exitCode != null ? ` · exit ${exitCode}` : ''}`
    }
    case 'webSearch': {
      const query = stringValue(input.query) ?? ''
      const output = outputObject(result)
      const sources = Array.isArray(output?.sources) ? output.sources.length : 0
      return `⌕ Search web for "${query}"${sources ? ` · ${sources} ${sources === 1 ? 'source' : 'sources'}` : ''}`
    }
    case 'webFetch': {
      return `↓ Fetch ${stringValue(input.url) ?? '(unknown url)'}`
    }
    case 'spawnWorker': {
      const agentId = outputObject(result)?.agentId ?? stringValue(input.agentId) ?? 'worker'
      const title = outputObject(result)?.title ?? stringValue(input.taskId) ?? ''
      return `▶ Start worker ${agentId}${title ? ` · ${title}` : ''}`
    }
    case 'skill': {
      return `▸ Activate skill ${stringValue(input.name) ?? '(unknown)'}`
    }
    default:
      return `· ${toolKey}`
  }
}

function renderRootOutcome(outcome: Extract<AppEvent, { type: 'turn_outcome' }>['outcome']): string | null {
  if (outcome._tag === 'Completed') return null
  return `✗ ${formatOutcome(outcome)}`
}

function renderWorkerOutcome(
  agent: AgentInfo | null,
  outcome: Extract<AppEvent, { type: 'turn_outcome' }>['outcome'],
  forkId: string,
): string {
  const label = agentLabel(agent, forkId)
  if (outcome._tag === 'Completed') {
    const count = outcome.completion.toolCallsCount
    return `✓ ${label} done · ${count} ${count === 1 ? 'tool' : 'tools'}`
  }
  if (outcome._tag === 'Cancelled') return `■ ${label} cancelled`
  return `✗ ${label} failed · ${formatOutcome(outcome)}`
}

function renderAgentStart(agent: AgentInfo): string {
  return `▶ ${agentLabel(agent, agent.forkId)} started · ${agent.name}`
}

function renderUserMessage(content: string, taskMode: boolean): string {
  return taskMode ? `▸ Task: ${content}` : `> ${content}`
}

function textFromParts(parts: readonly { readonly _tag: string; readonly text?: string }[]): string {
  return parts
    .filter((part) => part._tag === 'TextPart')
    .map((part) => part.text ?? '')
    .join('')
}

function agentForFork(forkId: string, agentsByFork: Map<string, AgentInfo>): AgentInfo | null {
  return agentsByFork.get(forkId) ?? null
}

function agentLabel(agent: AgentInfo | null, fallback: string): string {
  if (!agent) return `[${fallback}] (agent)`
  return `[${agent.agentId}] (${agent.role})`
}

function prefixed(forkId: string | null, line: string, agentsByFork: Map<string, AgentInfo>): string {
  if (forkId === null) return line
  const agent = agentForFork(forkId, agentsByFork)
  return agent ? `  ${agentLabel(agent, forkId)} ${line}` : `  ${line}`
}

function outputObject(result: ToolResult | undefined): Record<string, unknown> | null {
  const output = successOutput(result)
  return typeof output === 'object' && output !== null && !Array.isArray(output)
    ? output as Record<string, unknown>
    : null
}

function successOutput(result: ToolResult | undefined): unknown {
  const like = toolResult(result)
  return like?._tag === 'Success' ? like.output : undefined
}

function resultTag(result: ToolResult | undefined): ToolResultTag {
  return toolResult(result)?._tag ?? 'Success'
}

function toolResult(result: ToolResult | undefined): ToolResultLike | null {
  return result && typeof result === 'object' && '_tag' in result
    ? result as ToolResultLike
    : null
}

function toolField(result: ToolResult | undefined, field: string): unknown {
  return toolResult(result)?.[field]
}

function toolErrorMessage(result: ToolResult | undefined): string {
  const errorValue = toolField(result, 'error')
  if (errorValue instanceof Error) return errorValue.message
  const object = objectValue(errorValue)
  return stringValue(object?.message) ?? stringify(errorValue ?? 'error')
}

function toolIssueMessage(result: ToolResult | undefined): string {
  const issue = objectValue(toolField(result, 'issue'))
  return stringValue(issue?.message) ?? 'invalid input'
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null
    ? value as Record<string, unknown>
    : null
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

function stringify(value: unknown): string {
  if (typeof value === 'string') return value
  if (value instanceof Error) return value.message
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function formatOutcome(outcome: Extract<AppEvent, { type: 'turn_outcome' }>['outcome']): string {
  switch (outcome._tag) {
    case 'ProviderNotReady':
      return `provider not ready (${outcome.detail._tag})`
    case 'ConnectionFailure':
      return `connection failure (${outcome.detail._tag})`
    case 'UnexpectedError':
      return outcome.detail.message
    case 'SafetyStop':
      return outcome.reason._tag === 'Other' ? outcome.reason.message : `safety stop (${outcome.reason._tag})`
    case 'ContextWindowExceeded':
      return 'context window exceeded'
    case 'OutputTruncated':
      return 'output truncated'
    case 'ParseFailure':
      return `tool input parse failure: ${outcome.error.issue.message}`
    case 'ToolInputValidationFailure':
      return `tool input validation failed: ${outcome.issue.message}`
    case 'ToolExecutionError':
      return `tool execution failed: ${outcome.error.message}`
    case 'GateRejected':
      return `tool rejected: ${outcome.toolName}`
    case 'Overthinking':
      return `thinking exceeded ${outcome.limit} characters`
    case 'Cancelled':
      return `cancelled (${outcome.reason._tag})`
    case 'Completed':
      return 'completed'
    default:
      return `unknown outcome (${outcome._tag})`
  }
}

function outcomeWillContinue(outcome: Extract<AppEvent, { type: 'turn_outcome' }>['outcome']): boolean {
  if (outcome._tag === 'Completed' && outcome.completion.yieldTarget !== null) return false
  return (
    (outcome._tag === 'Completed' && outcome.completion.toolCallsCount > 0)
    || outcome._tag === 'ParseFailure'
    || outcome._tag === 'ToolInputValidationFailure'
    || outcome._tag === 'ToolExecutionError'
    || outcome._tag === 'GateRejected'
    || outcome._tag === 'ConnectionFailure'
    || outcome._tag === 'ContextWindowExceeded'
    || outcome._tag === 'Overthinking'
  )
}
