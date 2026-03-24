/**
 * Agent Loop — runs the real Magnitude agent system for builder-bench scenarios.
 *
 * Uses createCodingAgentClient with Docker-bridged tools registered as
 * an override for the lead agent. The real TurnController, Cortex,
 * and ExecutionManager drive the turn loop.
 */

import { Effect, Layer } from 'effect'
import ansis from 'ansis'
import {
  createCodingAgentClient,
  ChatPersistence,
  registerAgentDefinition,
  clearAgentOverrides,
  isStable,
  textParts,
  type SessionContext,
  type AppEvent,
  type ChatPersistenceService,
  type SlotUsage,
} from '@magnitudedev/agent'
import { getEvalProviderClient } from '../../provider-runtime'
import type { ModelSpec } from '../../types'
import type { ToolBridgeResult } from './tool-bridge'

let runCounter = 0

// =============================================================================
// Types
// =============================================================================

export interface AgentLoopOptions {
  taskPrompt: string
  modelSpec: ModelSpec
  toolBridge: ToolBridgeResult
  maxTurns: number
  /** Working directory inside the Docker container */
  workDir: string
  /** Short label for log prefixing (e.g. scenario ID) */
  label: string
}

export interface AgentLoopResult {
  /** Whether the agent called done() (turn decision was 'finish') */
  agentDone: boolean
  /** Number of LLM turns executed */
  turnCount: number
  /** Cumulative token/cost usage for the primary model slot */
  usage: SlotUsage
  /** Wall-clock time in ms */
  wallTimeMs: number
}

// =============================================================================
// Null Persistence (no-op — evals don't need session storage)
// =============================================================================

const nullPersistence: ChatPersistenceService = {
  loadEvents: () => Effect.succeed([]),
  persistNewEvents: () => Effect.void,
  getSessionMetadata: () => Effect.succeed({ chatName: 'bench', workingDirectory: '/workspace', gitBranch: null }),
  saveSessionMetadata: () => Effect.void,
  saveArtifact: () => Effect.void,
}

// =============================================================================
// Synthetic Session Context
// =============================================================================

function buildSessionContext(workDir: string): SessionContext {
  return {
    cwd: workDir,
    platform: 'linux',
    shell: 'bash',
    timezone: 'UTC',
    username: 'bench',
    fullName: null,
    git: null,
    folderStructure: '(Docker container)',
    agentsFile: null,
    skills: null,
  }
}

// =============================================================================
// Agent Loop
// =============================================================================

/**
 * Run the real Magnitude agent for a builder-bench scenario.
 *
 * 1. Registers Docker-backed agent definition as 'lead' override
 * 2. Creates a headless agent client (no persistence, synthetic session context)
 * 4. Sends the task prompt and waits for the agent to become stable
 * 5. Collects metrics and cleans up
 */
export async function runAgentLoop(options: AgentLoopOptions): Promise<AgentLoopResult> {
  const { taskPrompt, modelSpec, toolBridge, maxTurns, workDir, label } = options
  const startTime = Date.now()
  const runId = ++runCounter
  const prefix = ansis.dim(`[${runId}] ${label}`)
  const log = (msg: string) => console.log(`  ${prefix} ${msg}`)

  // Register overrides
  registerAgentDefinition('lead', toolBridge.agentDef)

  try {
    // Create headless agent client (restores provider runtime state internally)
    const persistenceLayer = Layer.succeed(ChatPersistence, nullPersistence)
    const client = await createCodingAgentClient({
      persistence: persistenceLayer,
      sessionContext: buildSessionContext(workDir),
    })

    // Configure LLM AFTER client creation — client bootstrap may restore the primary slot from stored config
    // createCodingAgentClient can overwrite the primary slot from stored config
    const providerClient = await getEvalProviderClient()
    const auth = await providerClient.auth.getAuth(modelSpec.provider)
    await providerClient.state.setSelection('primary', modelSpec.provider, modelSpec.model, auth ?? null, { persist: false })
    await providerClient.state.resetUsage('primary')

    // Track turn count and completion
    let turnCount = 0
    let agentDone = false

    // Live trace output
    client.onEvent((event: AppEvent) => {
      if (event.type === 'turn_started' && event.forkId === null) {
        log(ansis.dim(`── turn ${turnCount + 1} ──`))
      }

      if (event.type === 'tool_event' && event.forkId === null) {
        if (event.event._tag === 'ToolInputReady') {
          const inputStr = typeof event.event.input === 'string'
            ? (event.event.input as string).slice(0, 80)
            : JSON.stringify(event.event.input).slice(0, 80)
          log(`${ansis.cyan(event.toolKey)}${ansis.dim('(')}${ansis.dim(inputStr)}${ansis.dim(')')}`)
        }
        if (event.event._tag === 'ToolExecutionEnded' && event.event.result._tag === 'Error') {
          log(`  ${ansis.red('error:')} ${event.event.result.error.slice(0, 100)}`)
        }
      }

      if (event.type === 'turn_completed' && event.forkId === null) {
        turnCount++

        if (event.result.success && event.result.turnDecision === 'finish') {
          agentDone = true
        }

        if (turnCount >= maxTurns && !agentDone) {
          client.send({ type: 'interrupt', forkId: null } as AppEvent)
        }
      }

      if (event.type === 'turn_unexpected_error' && event.forkId === null) {
        log(ansis.red(`error: ${event.message}`))
      }
    })

    // Wait for root fork to become stable.
    // Must see unstable first — otherwise resolves before the turn starts.
    const stablePromise = new Promise<void>((resolve) => {
      let sawUnstable = false
      client.state.working.subscribeFork(null, (state) => {
        if (!sawUnstable) {
          if (!isStable(state)) sawUnstable = true
          return
        }
        if (isStable(state)) resolve()
      })
    })

    // Send the task prompt
    await client.send({
      type: 'user_message',
      forkId: null,
      content: textParts(typeof taskPrompt === 'string' ? taskPrompt : String(taskPrompt)),
      attachments: [],
      mode: 'text',
      synthetic: false,
      taskMode: false,
    })

    // Wait for agent to become stable (with absolute timeout)
    const ABSOLUTE_TIMEOUT_MS = 5 * 60 * 1000 // 5 minutes
    let timeoutId: Timer
    await Promise.race([
      stablePromise,
      new Promise<void>((resolve) => { timeoutId = setTimeout(resolve, ABSOLUTE_TIMEOUT_MS) }),
    ])
    clearTimeout(timeoutId!)

    // Collect usage and clean up
    const usage = await providerClient.state.getUsage('primary')
    await client.dispose()

    return {
      agentDone,
      turnCount,
      usage,
      wallTimeMs: Date.now() - startTime,
    }
  } finally {
    clearAgentOverrides()
  }
}
