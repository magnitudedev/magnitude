/**
 * Cortex Worker (Forked) — Native Paradigm
 *
 * Thin orchestrator: encode → send → decode (via TurnEngine) → publish events → publish outcome.
 *
 * Uses:
 *  - NativeModelResolver to get a NativeBoundModel
 *  - TurnEngine.runTurn to get a Stream<TurnEngineEvent>
 *  - liftTurnEngineEvent to promote TurnEngineEvents to AppEvents
 *  - persistResult to write tool results to disk
 *
 * xml-act path is orphaned. This worker is the sole turn handler.
 */

import { Effect, Stream, Layer } from 'effect'
import type { TurnEngineEvent, RegisteredTool } from '@magnitudedev/turn-engine'
import { Worker, AmbientServiceTag, WorkerBusTag, Fork } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import path from 'path'

import { Schema as EffectSchema } from '@effect/schema'
import * as JSONSchema from '@effect/schema/JSONSchema'

import type { AppEvent, TurnOutcome } from '../events'

import { MemoryProjection } from '../projections/memory'
import { SessionContextProjection } from '../projections/session-context'
import { AgentStatusProjection } from '../projections/agent-status'

import { TurnEngine } from '../engine/turn-engine'
import type { TurnEngineError } from '../engine/turn-engine'
import { makeToolRegistryLive } from '../engine/tool-registry'
import { NativeModelResolver } from '../engine/native-model-resolver'

import { liftTurnEngineEvent, resolveToolKey } from '../lift-engine-event'
import type { ToolKey } from '../catalog'
import type { ToolDef, ResponseUsage } from '@magnitudedev/codecs'
import type { TurnEngineOutcome } from '@magnitudedev/turn-engine'

import { buildRegisteredTools } from '../tools/tool-registry'
import { buildResolvedToolSet } from '../tools/resolved-toolset'
import { getAgentDefinition, getForkInfo } from '../agents/registry'
import { ExecutionManager } from '../execution/types'
import { ConfigAmbient } from '../ambient/config-ambient'
import { buildInterruptedTurnOutcome } from '../util/interrupt-utils'
import type { ObservationPart } from '@magnitudedev/roles'
import { TurnContextTag } from '../engine/turn-context'
import { persistResult } from '../runtime/result-persistence'

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
// Fold accumulator for a single turn
// =============================================================================

interface TurnAccumulator {
  toolCallsCount: number
  outcome: TurnEngineOutcome | null
  usage: ResponseUsage | null
  /** toolCallId → ToolKey, populated as engine events with toolName flow through.
   * Used by the lift to resolve toolKey for events that carry only toolCallId. */
  toolCallToToolKey: ReadonlyMap<string, ToolKey>
}

const initialAcc: TurnAccumulator = {
  toolCallsCount: 0,
  outcome: null,
  usage: null,
  toolCallToToolKey: new Map<string, ToolKey>(),
}

// =============================================================================
// Map TurnEngineOutcome → TurnOutcome
// =============================================================================

function mapEngineOutcomeToAgent(engineOutcome: TurnEngineOutcome | null): TurnOutcome {
  if (engineOutcome === null) {
    return { _tag: 'UnexpectedError', message: 'Stream ended without TurnEnd', detail: { _tag: 'EngineDefect' } }
  }
  switch (engineOutcome._tag) {
    case 'Completed':
      return {
        _tag: 'Completed',
        completion: {
          toolCallsCount: engineOutcome.toolCallsCount,
          finishReason: engineOutcome.toolCallsCount > 0 ? 'tool_calls' : 'stop',
          feedback: [],
        },
      }
    case 'OutputTruncated':
      return { _tag: 'OutputTruncated' }
    case 'ContentFiltered':
      return { _tag: 'UnexpectedError', message: 'Provider content filter triggered', detail: { _tag: 'ProviderDefect' } }
    case 'SafetyStop':
      return { _tag: 'SafetyStop', reason: engineOutcome.reason }
    case 'EngineDefect':
      return { _tag: 'UnexpectedError', message: engineOutcome.message, detail: { _tag: 'EngineDefect' } }
    case 'ToolInputDecodeFailure':
      return { _tag: 'UnexpectedError', message: `Tool input decode failure: ${String(engineOutcome.detail)}`, detail: { _tag: 'EngineDefect' } }
    case 'TurnStructureDecodeFailure':
      return { _tag: 'UnexpectedError', message: `Turn structure decode failure: ${String(engineOutcome.detail)}`, detail: { _tag: 'EngineDefect' } }
    case 'GateRejected':
      return { _tag: 'UnexpectedError', message: 'Gate rejected', detail: { _tag: 'CortexDefect' } }
    default: {
      const _ex: never = engineOutcome
      void _ex
      return { _tag: 'UnexpectedError', message: 'Unknown engine outcome', detail: { _tag: 'CortexDefect' } }
    }
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
        // 9. Drain stream: lift events, persist tool results, accumulate outcome
        // ──────────────────────────────────────────────────────────────────────
        const acc = yield* stream.pipe(
          Stream.runFoldEffect(initialAcc, (acc, event) => Effect.gen(function* () {
            // a) maintain toolCallId → ToolKey map: capture mapping when an
            //    engine event arrives that carries both toolCallId and toolName.
            //    Required so subsequent events that carry only toolCallId
            //    (ToolInputFieldChunk/Complete, ToolInputReady, ToolEmission)
            //    can be lifted with their toolKey resolved.
            let toolCallToToolKey = acc.toolCallToToolKey
            if (
              event._tag === 'ToolInputStarted'
              || event._tag === 'ToolExecutionStarted'
              || event._tag === 'ToolExecutionEnded'
              || event._tag === 'ToolInputDecodeFailure'
            ) {
              if (!toolCallToToolKey.has(event.toolCallId)) {
                const tk = resolveToolKey(
                  event.toolName,
                  registeredToolsMap as ReadonlyMap<string, RegisteredTool<unknown>>,
                )
                if (tk !== null) {
                  const next = new Map(toolCallToToolKey)
                  next.set(event.toolCallId, tk)
                  toolCallToToolKey = next
                }
              }
            }

            // b) lift to AppEvents and publish
            const appEvents = liftTurnEngineEvent(event, {
              forkId,
              turnId,
              registeredTools: registeredToolsMap as ReadonlyMap<string, RegisteredTool<unknown>>,
              toolCallToToolKey,
            })
            for (const ae of appEvents) {
              yield* publish(ae)
            }

            // c) count tool calls
            if (event._tag === 'ToolExecutionStarted') {
              return { ...acc, toolCallToToolKey, toolCallsCount: acc.toolCallsCount + 1 }
            }

            // c) persist successful tool results
            if (event._tag === 'ToolExecutionEnded' && event.result._tag === 'Success') {
              yield* persistResult(event.result.output, turnId, event.toolCallId, resultsDir).pipe(
                Effect.catchAll((e) => Effect.gen(function* () {
                  logger.warn({ forkId, turnId, toolCallId: event.toolCallId, e }, '[Cortex] persistResult failed')
                })),
              )
            }

            // d) capture outcome
            if (event._tag === 'TurnEnd') {
              return { ...acc, outcome: event.outcome, usage: event.usage }
            }

            return acc
          })),
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
            return null as TurnAccumulator | null
          })),
        )

        if (acc === null) return

        // ──────────────────────────────────────────────────────────────────────
        // 10. Publish turn_outcome
        // ──────────────────────────────────────────────────────────────────────
        const outcome = mapEngineOutcomeToAgent(acc.outcome)

        yield* publish({
          type: 'turn_outcome', forkId, turnId, chainId,
          strategyId: 'native',
          outcome,
          inputTokens:      acc.usage?.inputTokens ?? null,
          outputTokens:     acc.usage?.outputTokens ?? null,
          cacheReadTokens:  acc.usage?.cacheReadTokens ?? null,
          cacheWriteTokens: acc.usage?.cacheWriteTokens ?? null,
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
