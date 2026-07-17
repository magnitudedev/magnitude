/**
 * Cortex Worker (Forked) — Harness Paradigm
 *
 * Thin orchestrator: resolve model → build harness → run turn → publish events → publish outcome.
 *
 * Uses:
 *  - AgentModelResolver to resolve a model for the role
 *  - createHarness to stream model responses and dispatch tool execution
 *  - createHarnessAdapter to translate HarnessEvent → AppEvent
 */

import { Cause, Effect, Layer, Stream } from 'effect'
import { Worker, AmbientServiceTag } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import { createHarness } from '@magnitudedev/harness'
import type { AppEvent } from '../events'
import { finalizeModelAttemptFailure, type AgentStreamStartFailure } from '../errors'
import { describeThrown, stackTraceLines } from '../errors/formatters'

import { WindowProjection } from '../window'
import { SessionContextProjection } from '../projections/session-context'
import { AgentLifecycleProjection, getAgentByForkId } from '../projections/agent-lifecycle'
import { HarnessStateProjection } from '../projections/harness-state'
import { AgentToolkitProjection, readCoherentAgentToolkit } from '../projections/agent-toolkit'
import { TurnProjection, type ForkTurnState } from '../projections/turn'
import { MAX_RETRIES } from '../util/retry-backoff'

import { AgentModelResolver } from '../model/model-resolver'
import { getAgentDefinition, getForkInfo } from '../agents/registry'
import { materializeAgentToolkit } from '../tools/toolkits'
import { createHarnessAdapter } from '../execution/harness-adapter'
import { buildSystemPrompt } from '../prompts/system-prompt-builder'
import { windowToPrompt, createAgentFormatter } from '../prompts/window-to-prompt'
import { createToolResultFormatter } from '@magnitudedev/harness'

import { ShadowVcs } from '@magnitudedev/vcs'
import { ExecutionManager } from '../execution/types'
import { SkillsAmbient } from '../ambient/skills-ambient'
import { buildInterruptedTurnOutcome } from '../util/interrupt-utils'
import type { ContextPart } from '../content'
import type { ObservablePart } from '../observables/types'
import type { BaseCallOptions } from '@magnitudedev/sdk'
import { normalizeVision } from '@magnitudedev/ai'
import { captureContextImageInline, type ContextImageCaptureError } from '../util/capture-context-image'

function captureObservablePart(
  part: ObservablePart,
  scratchpadPath: string,
): Effect.Effect<ContextPart, ContextImageCaptureError> {
  switch (part._tag) {
    case 'TextPart':
      return Effect.succeed<ContextPart>({ _tag: 'ContextText', text: part.text })
    case 'ImagePart':
      return captureContextImageInline({ base64: part.data, mediaType: part.mediaType, scratchpadPath, name: 'observation' })
  }
}
import { isToolKey, type ToolKey } from '../tools/toolkits'


import { buildStandardHooks } from '../execution/harness-hooks'
import { TurnContextTag } from '../engine/turn-context'
import { ConfigAmbient } from '../ambient/config-ambient'
import { getSlotConfigForRole, getSlotConfigOrNull } from '../ambient/config-ambient'
import { ROLE_TO_SLOT } from '@magnitudedev/roles'
import { ImageQueryTarget } from '../tools/query-image'
import { SessionOptionsAmbient } from '../ambient/session-ambient'
import { ToolUniverseAmbient } from '../ambient/tool-universe-ambient'

function cortexDefectMessage(
  title: string,
  context: {
    readonly forkId: string | null
    readonly turnId: string
  },
  error: unknown,
): string {
  return [
    title,
    `forkId: ${context.forkId}`,
    `turnId: ${context.turnId}`,
    `error: ${describeThrown(error)}`,
    ...stackTraceLines(error),
  ].join('\n')
}

/** Build run-turn options when this exact leader turn claimed an advisor-required escalation. */
export function buildObserverEscalationRunOptions(turnState: ForkTurnState | undefined, turnId: string): BaseCallOptions | undefined {
  // TEMPORARILY DISABLED: Observer/Advisor escalation.
  // if (!turnRequiresAdvisor(turnState, turnId)) return undefined
  // return {
  //   toolChoice: {
  //     type: 'function' as const,
  //     function: { name: 'message_advisor' },
  //   },
  //   magnitudeAdditionalOptions: {
  //     turn_constraints: { message: 'forbid' },
  //   },
  // }
  return undefined
}

// =============================================================================
// Worker
// =============================================================================

export const Cortex = Worker.defineForked<AppEvent>()({
  name: 'Cortex',

  forkLifecycle: {
    activateOn: 'agent_created',
    completeOn: ['agent_killed', 'worker_user_killed', 'worker_idle_closed'],
  },

  eventHandlers: {
    worker_user_killed: (event) => Effect.gen(function* () {
      if (event.forkId === null) return
      return yield* Effect.interrupt
    }),

    worker_idle_closed: (event) => Effect.gen(function* () {
      if (event.forkId === null) return
      return yield* Effect.interrupt
    }),

    turn_started: (event, publish, read) => {
      const { forkId, turnId, chainId } = event

      return Effect.gen(function* () {
        // ──────────────────────────────────────────────────────────────────────
        // 1. Read projections
        // ──────────────────────────────────────────────────────────────────────
        const sessionCtx   = yield* read(SessionContextProjection)
        const agentState   = yield* read(AgentLifecycleProjection)
        const windowState  = yield* read(WindowProjection, forkId)
        const turnState    = yield* read(TurnProjection, forkId)
        const harnessState = yield* read(HarnessStateProjection, forkId)

        const runTurnOptions = forkId === null ? buildObserverEscalationRunOptions(turnState, turnId) : undefined

        const forkInfo = getForkInfo(agentState, forkId)
        if (!forkInfo) return

        const { roleId } = forkInfo
        const agentDef = getAgentDefinition(roleId)

        // ──────────────────────────────────────────────────────────────────────
        // 2. Resolve model
        // ──────────────────────────────────────────────────────────────────────
        const ambientService = yield* AmbientServiceTag
        const { config: configState, toolkit: toolkitState } = yield* readCoherentAgentToolkit(read, forkId)
        const modelResolver = yield* AgentModelResolver
        const agentId = forkId
          ? getAgentByForkId(agentState, forkId)?.agentId ?? '000000000000'
          : '000000000000'
        const activeSlot = getSlotConfigForRole(configState, roleId)
        const agentModel = yield* modelResolver.resolveSlotConfig(activeSlot, agentId, roleId)

        // ──────────────────────────────────────────────────────────────────────
        // 3. Observations
        // ──────────────────────────────────────────────────────────────────────
        const execManager = yield* ExecutionManager
        const observations: ContextPart[] = []
        const boundObs = execManager.getObservables(forkId)
        for (const obs of boundObs) {
          const parts = yield* obs.observe()
          observations.push(...yield* Effect.forEach(
            parts,
            part => captureObservablePart(part, sessionCtx.context?.scratchpadPath ?? process.cwd()),
            { concurrency: 'unbounded' },
          ))
        }
        if (observations.length > 0) {
          yield* publish({ type: 'observations_captured', forkId, turnId, parts: observations })
        }

        // ──────────────────────────────────────────────────────────────────────
        // 4. Get toolkit and fork layer
        // ──────────────────────────────────────────────────────────────────────
        const sessionOptions = ambientService.getValue(SessionOptionsAmbient)
        const headless = sessionOptions.headless
        const forkLayer = execManager.getForkLayer(forkId)
        const universe = ambientService.getValue(ToolUniverseAmbient)
        const toolkit = materializeAgentToolkit(universe, toolkitState.toolKeys)
        if (!forkLayer) {
          const message = [
            'Cortex defect: fork layer not initialized',
            `forkId: ${forkId}`,
            `turnId: ${turnId}`,
          ].join('\n')
          logger.error({ forkId, turnId, message }, '[Cortex] Fork layer not initialized')
          yield* publish({
            type: 'turn_outcome', forkId, turnId, chainId,
            strategyId: 'native',
            outcome: { _tag: 'UnexpectedError', detail: { _tag: 'CortexDefect', message }, requestId: null },
            commitPolicy: { _tag: 'commitErrorOnly' },
            inputTokens: null, outputTokens: null,
            cacheReadTokens: null, cacheWriteTokens: null,
            cost: null,
            providerId: agentModel.providerId, modelId: agentModel.modelId,
          })
          return
        }

        const turnContextLayer = Layer.succeed(TurnContextTag, { turnId, chainId, forkId })
        const activeSlotId = ROLE_TO_SLOT[roleId]
        const otherSlot = getSlotConfigOrNull(configState, activeSlotId === 'primary' ? 'secondary' : 'primary')
        const turnLayer = Layer.merge(Layer.merge(forkLayer, turnContextLayer), Layer.succeed(ImageQueryTarget, { slot: otherSlot }))

        // Record turn-start checkpoint — captures state at the turn boundary
        // so checkpoint_rollback can restore to "before this turn".
        logger.info({ forkId, turnId }, '[Cortex] Recording turn-start checkpoint')
        yield* Effect.gen(function* () {
          const vcs = yield* ShadowVcs
          yield* vcs.record({ message: `turn-start:${turnId}` })
        }).pipe(
          Effect.provide(turnLayer),
          Effect.catchAllCause((cause) => {
            logger.error({ forkId, turnId, cause }, '[Cortex] turn-start checkpoint failed')
            return Effect.void
          }),
        )

        // ──────────────────────────────────────────────────────────────────────
        // 5. Build system prompt
        // ──────────────────────────────────────────────────────────────────────
        const skills = ambientService.getValue(SkillsAmbient)

        const scratchpadPath = sessionCtx.context?.scratchpadPath ?? process.cwd()

        // Pass engine state for crash recovery — allows the harness to skip
        // tools that already executed before the process crashed.
        const engineState = harnessState?.engine
        const hasRecoverableState = engineState && engineState.toolOutcomes.size > 0

        const harness = createHarness({
          model: agentModel.model,
          toolkit,
          layer: turnLayer,
          initialState: hasRecoverableState ? engineState : undefined,
          maxThoughtChars: agentDef.maxThoughtChars,
          hooks: buildStandardHooks({
            forkId,
            turnId,
            agentDef,
            scratchpadPath,
          }),
        })

        const systemPrompt = buildSystemPrompt({
          roleDef: agentDef,
          skills,
          vcsAvailable: sessionOptions.vcsAvailable,
          headless,
          systemPromptOverride: sessionOptions.systemPromptOverride,
        })

        const timezone = sessionCtx.context?.timezone ?? null
        const formatter = createAgentFormatter(createToolResultFormatter(toolkit), { includeImageData: activeSlot.vision === true })

        const rawPrompt = windowToPrompt({
          windowState,
          systemPrompt,
          timezone,
          formatter,
          autopilotEnabled: windowState.autopilotEnabled,
          leaderLastAutopilotKnowledge: windowState.consumerAutopilotKnowledge.leader,
          includeImageData: activeSlot.vision === true,
        })
        const prompt = activeSlot.vision === true
          ? rawPrompt
          : normalizeVision(rawPrompt, () => '')

        // With the file-based attachment system, images are stored as files in
        // the session scratchpad. The agent reads them with its existing file
        // tools (read_file, query_image). No client-side image description.

        // ──────────────────────────────────────────────────────────────────────
        // 7. Build adapter
        // ──────────────────────────────────────────────────────────────────────
        const agentKind = agentDef.agentKind
        const defaultProseDest = agentKind === 'worker'
          ? { kind: 'coordinator' as const }
          : { kind: 'user' as const }

        // Build toolName → ToolKey map from toolkit
        const toolNameToKey = new Map<string, ToolKey>()
        for (const key of toolkit.keys) {
          if (isToolKey(key)) {
            const entry = toolkit.entries[key]
            const toolName = entry.tool.definition.name
            toolNameToKey.set(toolName, key as ToolKey)
          }
        }

        const adapter = createHarnessAdapter({
          forkId,
          turnId,
          chainId,
          roleId,
          defaultProseDest,
          publish,
          identicalResponseTracker: null,
          retryCount: (yield* read(TurnProjection, forkId))?.connectionRetryCount ?? 0,
          maxRetries: MAX_RETRIES,
          resolveToolKey: (toolName: string) => toolNameToKey.get(toolName),
        })

        // ──────────────────────────────────────────────────────────────────────
        // 8. Run turn
        // ──────────────────────────────────────────────────────────────────────
        const liveTurn = yield* harness.runTurn(prompt, runTurnOptions).pipe(
          Effect.provide(turnLayer),
          Effect.catchAll((err: AgentStreamStartFailure) => Effect.gen(function* () {
            logger.error({ forkId, turnId, err }, '[Cortex] Pre-stream failure')
            const turnFork = yield* read(TurnProjection, forkId)
            const decision = finalizeModelAttemptFailure({
              failure: err,
              retryCount: turnFork?.connectionRetryCount ?? 0,
              maxRetries: MAX_RETRIES,
            })

            yield* publish({
              type: 'turn_outcome', forkId, turnId, chainId,
              strategyId: 'native',
              outcome: decision.outcome,
              commitPolicy: decision.commitPolicy,
              inputTokens: null, outputTokens: null,
              cacheReadTokens: null, cacheWriteTokens: null,
              cost: null,
              providerId: agentModel.providerId, modelId: agentModel.modelId,
            })
            return null
          })),
        )

        if (liveTurn === null) return

        // ──────────────────────────────────────────────────────────────────────
        // 9. Consume events via adapter
        // ──────────────────────────────────────────────────────────────────────
        yield* Stream.runForEach(liveTurn.events, (event) => adapter.processEvent(event))

        // ──────────────────────────────────────────────────────────────────────
        // 9.5 Record turn-end checkpoint
        // ──────────────────────────────────────────────────────────────────────
        logger.info({ forkId, turnId }, '[Cortex] Recording turn-end checkpoint')
        yield* Effect.gen(function* () {
          const vcs = yield* ShadowVcs
          yield* vcs.record({ message: `turn-end:${turnId}` })
        }).pipe(
          Effect.provide(turnLayer),
          Effect.catchAllCause((cause) => {
            logger.error({ forkId, turnId, cause }, '[Cortex] turn-end checkpoint failed')
            return Effect.void
          }),
        )

        // ──────────────────────────────────────────────────────────────────────
        // 10. Publish turn_outcome
        // ──────────────────────────────────────────────────────────────────────
        const executeResult = adapter.getResult()

        yield* publish({
          type: 'turn_outcome', forkId, turnId, chainId,
          strategyId: 'native',
          outcome: executeResult.result,
          commitPolicy: executeResult.commitPolicy ?? (
            executeResult.result._tag === 'Completed'
              ? { _tag: 'commitCleanTurn' }
              : { _tag: 'commitErrorOnly' }
          ),
          inputTokens:      executeResult.usage?.inputTokens ?? null,
          outputTokens:     executeResult.usage?.outputTokens ?? null,
          cacheReadTokens:  executeResult.usage?.cacheReadTokens ?? null,
          cacheWriteTokens: executeResult.usage?.cacheWriteTokens ?? null,
          cost:             executeResult.usage?.cost ?? null,
          providerId: agentModel.providerId,
          modelId:    agentModel.modelId,
        })
      }).pipe(
        Effect.onInterrupt(() => Effect.gen(function* () {
          const turnOutcome = yield* buildInterruptedTurnOutcome({ forkId, turnId, chainId })
          yield* publish(turnOutcome)
        }).pipe(Effect.orDie)),
        Effect.catchAllCause((cause) => Effect.gen(function* () {
          const message = cortexDefectMessage('Cortex defect while handling turn_started', { forkId, turnId }, Cause.pretty(cause))
          logger.error({ context: 'Cortex', forkId, turnId, message, cause: Cause.pretty(cause) }, '[Cortex] Unexpected error in turn_started')
          yield* publish({
            type: 'turn_outcome', forkId, turnId, chainId,
            strategyId: 'native',
            outcome: { _tag: 'UnexpectedError', detail: { _tag: 'CortexDefect', message }, requestId: null },
            commitPolicy: { _tag: 'commitErrorOnly' },
            inputTokens: null, outputTokens: null,
            cacheReadTokens: null, cacheWriteTokens: null,
            cost: null,
            providerId: null, modelId: null,
          })
        })),
      )
    },
  },
})
