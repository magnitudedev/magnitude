/**
 * Cortex Worker (Forked) — Native Paradigm
 *
 * Thin orchestrator: encode → send → decode (via TurnEngine) → publish events → publish outcome.
 *
 * Uses:
 *  - NativeModelResolver to get a NativeBoundModel
 *  - TurnEngine.runTurn to get a Stream<TurnEngineEvent>
 *  - ExecutionManager.execute() to process the event stream with full policy
 *
 * xml-act path is orphaned. This worker is the sole turn handler.
 */

import { Effect, Stream, Layer } from 'effect'
import type { TurnEngineEvent } from '@magnitudedev/turn-engine'
import { Worker, AmbientServiceTag, WorkerBusTag, Fork } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import path from 'path'

import { Schema as EffectSchema } from '@effect/schema'
import * as JSONSchema from '@effect/schema/JSONSchema'

import type { AppEvent, TurnOutcome } from '../events'

import { MemoryProjection } from '../projections/memory'
import { ReplayProjection } from '../projections/replay'
import { SessionContextProjection } from '../projections/session-context'
import { AgentStatusProjection } from '../projections/agent-status'

import { TurnEngine } from '../engine/turn-engine'
import type { TurnEngineError } from '../engine/turn-engine'

import { NativeModelResolver } from '../engine/native-model-resolver'

import type { ToolDef } from '@magnitudedev/codecs'
import type { TurnEvent, TurnEventSink } from '../execution/types'

import { buildRegisteredTools } from '../tools/tool-registry'
import { buildResolvedToolSet } from '../tools/resolved-toolset'
import { getAgentDefinition, getForkInfo } from '../agents/registry'
import { ExecutionManager } from '../execution/types'
import type { ExecuteResult } from '../execution/types'
import { ConfigAmbient } from '../ambient/config-ambient'
import { buildInterruptedTurnOutcome } from '../util/interrupt-utils'
import type { ObservationPart } from '@magnitudedev/roles'
import { TurnContextTag } from '../engine/turn-context'

// =============================================================================
// Helpers
// =============================================================================

function deriveToolJsonSchema(schema: EffectSchema.Schema.Any): unknown {
  try {
    return JSONSchema.make(schema)
  } catch {
    return { type: 'object', properties: {}, additionalProperties: true }
  }
}


// =============================================================================
// Map TurnEngineError → TurnOutcome
// =============================================================================

function mapEngineErrorToOutcome(err: TurnEngineError): TurnOutcome {
  switch (err.phase) {
    case 'encode': return { _tag: 'UnexpectedError', message: err.message, detail: { _tag: 'EngineDefect' } }
    case 'send':   return { _tag: 'ConnectionFailure', detail: { _tag: 'TransportError' } }
    case 'decode': return { _tag: 'UnexpectedError', message: err.message, detail: { _tag: 'EngineDefect' } }
    case 'engine': return { _tag: 'UnexpectedError', message: err.message, detail: { _tag: 'EngineDefect' } }
    default:       return { _tag: 'UnexpectedError', message: err.message, detail: { _tag: 'CortexDefect' } }
  }
}

// =============================================================================
// Worker
// =============================================================================

export const Cortex = Worker.defineForked<AppEvent>()({
  name: 'Cortex',

  forkLifecycle: {
    activateOn: 'agent_created',
    completeOn: ['agent_killed', 'subagent_user_killed', 'subagent_idle_closed'],
  },

  eventHandlers: {
    subagent_user_killed: (event) => Effect.gen(function* () {
      if (event.forkId === null) return
      return yield* Effect.interrupt
    }),

    subagent_idle_closed: (event) => Effect.gen(function* () {
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
        const agentState   = yield* read(AgentStatusProjection)
        const memoryState  = yield* read(MemoryProjection, forkId)
        const replayState  = yield* read(ReplayProjection, forkId)

        const forkInfo = getForkInfo(agentState, forkId)
        if (!forkInfo) return

        const { variant, slot: modelSlot } = forkInfo
        const agentDef = getAgentDefinition(variant)

        // ──────────────────────────────────────────────────────────────────────
        // 2. Resolve native model
        // ──────────────────────────────────────────────────────────────────────
        const modelResolver = yield* NativeModelResolver
        const resolveResult = yield* modelResolver.resolve(modelSlot).pipe(Effect.either)

        if (resolveResult._tag === 'Left') {
          const err = resolveResult.left
          logger.warn({ forkId, turnId, err }, '[Cortex] NativeModelResolver failed — publishing ProviderNotReady')
          yield* publish({
            type:    'turn_outcome',
            forkId, turnId, chainId,
            strategyId: 'native',
            outcome: { _tag: 'ProviderNotReady', detail: { _tag: 'NotConfigured' } },
            inputTokens: null, outputTokens: null,
            cacheReadTokens: null, cacheWriteTokens: null,
            providerId: null, modelId: null,
          })
          return
        }
        const boundModel = resolveResult.right

        // ──────────────────────────────────────────────────────────────────────
        // 3. Observations
        // ──────────────────────────────────────────────────────────────────────
        const execManager   = yield* ExecutionManager
        const observations: ObservationPart[] = []
        const boundObs = execManager.getObservables(forkId)
        for (const obs of boundObs) {
          const parts = yield* obs.observe()
          observations.push(...parts)
        }
        if (observations.length > 0) {
          yield* publish({ type: 'observations_captured', forkId, turnId, parts: observations })
        }

        // ──────────────────────────────────────────────────────────────────────
        // 4. Build tool registry
        // ──────────────────────────────────────────────────────────────────────
        const ambientService = yield* AmbientServiceTag
        const configState    = ambientService.getValue(ConfigAmbient)

        const toolSet = buildResolvedToolSet(agentDef, configState, modelSlot)

        const workerBus = yield* WorkerBusTag<AppEvent>()
        const forkLayer = execManager.getForkLayer(forkId)
        if (!forkLayer) {
          logger.error({ forkId, turnId }, '[Cortex] Fork layer not initialized — aborting turn')
          yield* publish({
            type: 'turn_outcome', forkId, turnId, chainId,
            strategyId: 'native',
            outcome: { _tag: 'UnexpectedError', message: 'Fork layer not initialized', detail: { _tag: 'CortexDefect' } },
            inputTokens: null, outputTokens: null,
            cacheReadTokens: null, cacheWriteTokens: null,
            providerId: boundModel.model.providerId, modelId: boundModel.model.id,
          })
          return
        }
        const toolDILayer = Layer.mergeAll(
          forkLayer,
          Layer.succeed(WorkerBusTag<AppEvent>(), workerBus),
          Layer.succeed(TurnContextTag, { turnId, forkId }),
        )
        const registeredToolsMap = buildRegisteredTools(toolSet, toolDILayer)
        const registeredTools    = Array.from(registeredToolsMap.values())

        // ──────────────────────────────────────────────────────────────────────
        // 5. Build toolDefs
        // ──────────────────────────────────────────────────────────────────────
        const toolDefs: ToolDef[] = registeredTools.map(rt => ({
          name:        rt.toolName,
          description: rt.tool.description ?? '',
          parameters:  deriveToolJsonSchema(rt.tool.inputSchema),
        }))

        // ──────────────────────────────────────────────────────────────────────
        // 6. Persistence directory
        // ──────────────────────────────────────────────────────────────────────
        // TODO: expose resultsDir from config/session context to avoid hardcoding
        if (!sessionCtx.context?.workspacePath) {
          logger.warn({ forkId, turnId }, '[Cortex] No session context — falling back to process.cwd() for results dir')
        }
        const workspacePath = sessionCtx.context?.workspacePath ?? process.cwd()
        const resultsDir = path.join(workspacePath, '.results')

        // ──────────────────────────────────────────────────────────────────────
        // 7. Message destination based on agent role
        // ──────────────────────────────────────────────────────────────────────
        const messageDestination = variant === 'worker' ? 'parent' : 'user'

        // ──────────────────────────────────────────────────────────────────────
        // 8. Run turn via TurnEngine
        // ──────────────────────────────────────────────────────────────────────
        const turnEngine = yield* TurnEngine

        type EngineStream = Stream.Stream<TurnEngineEvent, TurnEngineError>

        const stream: EngineStream | null = yield* turnEngine.runTurn({
          model:              boundModel,
          memory:             memoryState.messages,
          tools:              registeredToolsMap,
          toolDefs,
          options:            { thinkingLevel: 'medium' },
          messageDestination,
          initialState:       replayState,
        }).pipe(
          Effect.catchTag('TurnEngineError', (err: TurnEngineError) => Effect.gen(function* () {
            logger.error({ forkId, turnId, err }, '[Cortex] TurnEngine pre-stream error')
            yield* publish({
              type: 'turn_outcome', forkId, turnId, chainId,
              strategyId: 'native',
              outcome: mapEngineErrorToOutcome(err),
              inputTokens: null, outputTokens: null,
              cacheReadTokens: null, cacheWriteTokens: null,
              providerId: boundModel.model.providerId, modelId: boundModel.model.id,
            })
            return null as EngineStream | null
          })),
        )

        if (stream === null) return

        // ──────────────────────────────────────────────────────────────────────
        // 9. Execute turn via ExecutionManager
        // ──────────────────────────────────────────────────────────────────────
        const sink: TurnEventSink = {
          emit: (event: TurnEvent) => Effect.gen(function* () {
            switch (event._tag) {
              case 'ThinkingStart':
                yield* publish({ type: 'thinking_start', forkId, turnId })
                break
              case 'ThinkingDelta':
                yield* publish({ type: 'thinking_chunk', forkId, turnId, text: event.text })
                break
              case 'ThinkingEnd':
                yield* publish({ type: 'thinking_end', forkId, turnId })
                break
              case 'MessageStart':
                yield* publish({ type: 'message_start', forkId, turnId, id: event.id, destination: event.destination })
                break
              case 'MessageChunk':
                yield* publish({ type: 'message_chunk', forkId, turnId, id: event.id, text: event.text })
                break
              case 'MessageEnd':
                yield* publish({ type: 'message_end', forkId, turnId, id: event.id })
                break
              case 'ToolEvent':
                yield* publish({ type: 'tool_event', forkId, turnId, toolCallId: event.toolCallId, toolKey: event.toolKey, event: event.event })
                break
              case 'RawResponseChunk':
                yield* publish({ type: 'raw_response_chunk', forkId, turnId, text: event.text })
                break
              case 'TurnResult':
                // Terminal — handled via execute() return value
                break
            }
          }),
        }

        const executeResult: ExecuteResult | null = yield* execManager.execute(stream, {
          forkId,
          turnId,
          chainId,
          defaultProseDest: messageDestination as 'user' | 'parent',
          triggeredByUser: event.chainId === event.turnId, // first turn in chain = user-triggered
          toolSet,
        }, sink).pipe(
          Effect.catchTag('TurnEngineError', (err: TurnEngineError) => Effect.gen(function* () {
            logger.error({ forkId, turnId, err }, '[Cortex] TurnEngine mid-stream error')
            yield* publish({
              type: 'turn_outcome', forkId, turnId, chainId,
              strategyId: 'native',
              outcome: mapEngineErrorToOutcome(err),
              inputTokens: null, outputTokens: null,
              cacheReadTokens: null, cacheWriteTokens: null,
              providerId: boundModel.model.providerId, modelId: boundModel.model.id,
            })
            return null
          })),
        )

        if (executeResult === null) return

        // ──────────────────────────────────────────────────────────────────────
        // 10. Publish turn_outcome
        // ──────────────────────────────────────────────────────────────────────
        yield* publish({
          type: 'turn_outcome', forkId, turnId, chainId,
          strategyId: 'native',
          outcome: executeResult.result,
          inputTokens:      executeResult.usage?.inputTokens ?? null,
          outputTokens:     executeResult.usage?.outputTokens ?? null,
          cacheReadTokens:  executeResult.usage?.cacheReadTokens ?? null,
          cacheWriteTokens: executeResult.usage?.cacheWriteTokens ?? null,
          providerId: boundModel.model.providerId,
          modelId:    boundModel.model.id,
        })
      }).pipe(
        Effect.onInterrupt(() => Effect.gen(function* () {
          const turnOutcome = yield* buildInterruptedTurnOutcome({ forkId, turnId, chainId })
          yield* publish(turnOutcome)
        }).pipe(Effect.orDie)),
        Effect.catchAll((error: unknown) => Effect.gen(function* () {
          const message = error instanceof Error ? error.message : String(error)
          logger.error({ context: 'Cortex', forkId, turnId, error: message }, '[Cortex] Unexpected error in turn_started')
          yield* publish({
            type: 'turn_outcome', forkId, turnId, chainId,
            strategyId: 'native',
            outcome: { _tag: 'UnexpectedError', message, detail: { _tag: 'CortexDefect' } },
            inputTokens: null, outputTokens: null,
            cacheReadTokens: null, cacheWriteTokens: null,
            providerId: null, modelId: null,
          })
        })),
      )
    },
  },
})
