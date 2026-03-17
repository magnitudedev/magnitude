import { Effect, Layer } from 'effect'
import {
  ChatPersistence,
  collectSessionContext,
  createCodingAgentClient,
  type AppEvent,
} from '@magnitudedev/agent'
import { ProviderAuth, ProviderState, makeProviderRuntimeLive } from '@magnitudedev/providers'
import { createStorageClient } from '@magnitudedev/storage'
import { initLogger } from '@magnitudedev/logger'
import { JsonChatPersistence } from './persistence'
import ansis from 'ansis'

// =============================================================================
// Output helpers
// =============================================================================

const lbl = ansis.hex('#0ea5e9').bold   // blue.500 — primary
const dim = ansis.hex('#94a3b8')         // slate.400 — muted
const ok = ansis.hex('#1f9670').bold     // green.600 — success
const err = ansis.hex('#f87171').bold    // red.400 — error
const agt = ansis.hex('#c4b5fd')         // violet.300 — modePlan

let _flushMessage: (() => void) | null = null

function line(msg: string, indent = 0) {
  if (_flushMessage) _flushMessage()
  const prefix = indent > 0 ? '  '.repeat(indent) : ''
  process.stdout.write(`${prefix}${msg}\n`)
}

// =============================================================================
// Oneshot entry point
// =============================================================================

export interface RunOneshotOptions {
  prompt?: string
  providerId?: string
  modelId?: string
  debug?: boolean
  disableShellSafeguards?: boolean
  disableCwdSafeguards?: boolean
}

export async function runOneshot(options: RunOneshotOptions): Promise<void> {
  const {
    prompt,
    providerId,
    modelId,
    debug,
    disableShellSafeguards,
    disableCwdSafeguards,
  } = options

  if (!prompt) {
    console.error('A prompt is required for oneshot mode')
    process.exit(1)
  }
  if (!providerId) {
    console.error('--provider is required for oneshot mode')
    process.exit(1)
  }
  if (!modelId) {
    console.error('--model is required for oneshot mode')
    process.exit(1)
  }

  process.stdout.write(`\n${lbl('magnitude')} ${dim('oneshot')}\n`)
  line(`${dim('provider')} ${providerId}  ${dim('model')} ${modelId}`)
  line('')

  const storage = await createStorageClient({ cwd: process.cwd() })
  const providerRuntime = makeProviderRuntimeLive()

  await Effect.runPromise(
    Effect.gen(function* () {
      const auth = yield* ProviderAuth
      const state = yield* ProviderState

      const authStatus = yield* auth.detectProviderAuthMethods(providerId)
      const usable = authStatus?.methods.find((method) => method.connected)

      if (!usable) {
        throw new Error(
          `Provider '${providerId}' has no valid auth. Set the appropriate env var for that provider.`,
        )
      }

      const selectedAuth = usable.auth ?? null

      yield* state.setSelection('primary', providerId, modelId, selectedAuth, { persist: false })
      yield* state.setSelection('secondary', providerId, modelId, selectedAuth, { persist: false })
      yield* state.setSelection('browser', providerId, modelId, selectedAuth, { persist: false })
    }).pipe(Effect.provide(providerRuntime)),
  ).catch((e) => {
    line(`✗ ${err(e instanceof Error ? e.message : String(e))}`)
    process.exit(1)
  })

  line(`✓ ${dim('auth detected')}`)

  const persistence = new JsonChatPersistence({
    storage,
    workingDirectory: process.cwd(),
  })

  initLogger(persistence.getSessionId())

  const sessionContext = await collectSessionContext({
    cwd: process.cwd(),
    memoryEnabled: false,
    storage,
    oneshot: { prompt },
  })

  const persistenceLayer = Layer.succeed(ChatPersistence, persistence)

  const client = await createCodingAgentClient({
    persistence: persistenceLayer,
    storage,
    debug,
    sessionContext,
    providerRuntime,
    disableShellSafeguards,
    disableCwdSafeguards,
  })

  line(`✓ ${dim('session initialized')}`)
  line('')

  await client.send({
    type: 'oneshot_task',
    forkId: null,
    prompt,
  })

  // ---------------------------------------------------------------------------
  // Exit handling
  // ---------------------------------------------------------------------------

  let settled = false
  const exit = async (code: number) => {
    if (settled) return
    settled = true
    line('')
    line(code === 0 ? `✓ ${ok('finished')}` : `✗ ${err(`exiting with code ${code}`)}`)
    try { await client.dispose() } finally { process.exit(code) }
  }

  // ---------------------------------------------------------------------------
  // State for output formatting
  // ---------------------------------------------------------------------------

  const agentNames = new Map<string, string>()
  const agentRoles = new Map<string, string>()
  const activeLensContent = new Map<string | null, string>()
  const pendingAgents: Array<{ forkId: string; role: string; name: string }> = []
  const shownAgents = new Set<string>()
  let lastForkId: string | null | undefined = undefined
  let messageLineOpen = false

  _flushMessage = () => {
    if (messageLineOpen) {
      process.stdout.write('\n')
      messageLineOpen = false
    }
  }

  function showAgentHeader(forkId: string, role: string, name: string) {
    line(agt(role) + dim(` › ${name}`))
    lastForkId = forkId
    shownAgents.add(forkId)
  }

  function flushPendingAgents() {
    for (const a of pendingAgents) {
      showAgentHeader(a.forkId, a.role, a.name)
    }
    pendingAgents.length = 0
  }

  function ensureAgentVisible(forkId: string) {
    // If this agent is pending, flush up to and including it
    const idx = pendingAgents.findIndex(a => a.forkId === forkId)
    if (idx !== -1) {
      for (let i = 0; i <= idx; i++) {
        const a = pendingAgents[i]
        showAgentHeader(a.forkId, a.role, a.name)
      }
      pendingAgents.splice(0, idx + 1)
    } else if (forkId !== lastForkId) {
      // Context switch — re-show header
      const name = agentNames.get(forkId) ?? forkId
      const role = agentRoles.get(forkId) ?? 'agent'
      showAgentHeader(forkId, role, name)
    }
  }

  function formatTurnInfo(event: Extract<AppEvent, { type: 'turn_completed' }>): string {
    const tokens = event.outputTokens
    const toolCounts = new Map<string, number>()
    for (const tc of event.toolCalls ?? []) {
      const name = tc.group ? `${tc.group}:${tc.toolName}` : tc.toolName
      toolCounts.set(name, (toolCounts.get(name) ?? 0) + 1)
    }
    const toolSummary = [...toolCounts.entries()]
      .map(([n, c]) => c > 1 ? `${n}×${c}` : n)
      .join(', ')
    return [toolSummary || null, tokens ? `${tokens} tok` : null]
      .filter(Boolean).join(' · ')
  }

  // ---------------------------------------------------------------------------
  // Event handler
  // ---------------------------------------------------------------------------

  client.onEvent((event: AppEvent) => {
    switch (event.type) {
      case 'turn_started':
        break

      case 'lens_start':
        activeLensContent.set(event.forkId, '')
        break

      case 'lens_chunk':
        activeLensContent.set(event.forkId, (activeLensContent.get(event.forkId) ?? '') + event.text)
        break

      case 'lens_end': {
        const content = (activeLensContent.get(event.forkId) ?? '').trim()
        line(dim(`[${event.name}] ${content}`))
        activeLensContent.delete(event.forkId)
        break
      }

      case 'turn_completed': {
        const isRoot = event.forkId === null
        const info = formatTurnInfo(event)

        if (isRoot) {
          if (info) line(dim(info))
          flushPendingAgents()
        } else {
          ensureAgentVisible(event.forkId!)
          if (info) line(dim(info), 1)
        }

        if (isRoot && event.result.success && event.result.turnDecision === 'finish') {
          void exit(0)
          return
        }

        break
      }

      case 'agent_created':
        agentNames.set(event.forkId, event.name)
        agentRoles.set(event.forkId, event.role)
        pendingAgents.push({ forkId: event.forkId, role: event.role, name: event.name })
        break

      case 'agent_dismissed':
        if (lastForkId === event.forkId) lastForkId = undefined
        agentNames.delete(event.forkId)
        agentRoles.delete(event.forkId)
        shownAgents.delete(event.forkId)
        break

      case 'message_chunk':
        if (event.forkId === null) {
          for (let i = 0; i < event.text.length; i++) {
            if (!messageLineOpen) {
              messageLineOpen = true
            }
            process.stdout.write(event.text[i])
            if (event.text[i] === '\n') {
              messageLineOpen = false
            }
          }
        }
        break

      case 'message_end':
        if (event.forkId === null && messageLineOpen) {
          process.stdout.write('\n')
          messageLineOpen = false
        }
        break

      case 'turn_unexpected_error':
        line(`✗ ${err(`UNEXPECTED ERROR: ${event.message}`)}`)
        if (event.forkId === null) {
          void exit(1)
          return
        }
        break
    }
  })

  client.onError((e) => {
    line(`✗ ${err(`framework error: ${e._tag}`)}`)
  })

  process.once('SIGINT', () => void exit(130))
  process.once('SIGTERM', () => void exit(130))

}