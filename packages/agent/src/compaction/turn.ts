/**
 * Agentic compaction turn — runs a normal agent turn for reflection/summarization.
 */

import { Data, Effect, Layer, Option, Stream, Ref } from 'effect'

import { AmbientServiceTag } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import { createHarness } from '@magnitudedev/harness'

import type { AppEvent } from '../events'

import type { ForkWindowState } from '../window'
import type { CompletedTurn } from '../window/types'
import { SessionContextProjection } from '../projections/session-context'
import { AgentModelResolver } from '../model/model-resolver'
import { getAgentByForkId } from '../projections/agent-lifecycle'
import { getAgentDefinition } from '../agents/registry'
import { getEffectiveToolkit } from '../tools/toolkits'
import { buildSystemPrompt } from '../prompts/system-prompt-builder'
import { buildCompactionPrompt } from './prompt'
import { createToolResultFormatter } from '@magnitudedev/harness'
import { createAgentFormatter, windowToPrompt } from '../prompts/window-to-prompt'

import { CompactionContextTag, type CompactResult } from './context'
import { computeCompactionSizing } from './estimate'

import { ExecutionManager } from '../execution/types'
import { SkillsAmbient } from '../ambient/skills-ambient'
import { buildStandardHooks } from '../execution/harness-hooks'
import type { RoleId } from '../agents/role-validation'
import { COMPACTION_MAX_RETRIES } from '../constants'
import type { AgentLifecycleState } from '../projections/agent-lifecycle'
import { ConfigAmbient } from '../ambient/config-ambient'
import { SessionOptionsAmbient } from '../ambient/session-ambient'

class CompactionTurnError extends Data.TaggedError('CompactionTurnError')<{
  readonly reason: 'ForkLayerMissing' | 'EmptyResponse'
  readonly message: string
}> {}
import { createId } from '../util/id'
import { AgentModelOperationContextTag } from '../model/agent-model'
import { TurnContextTag } from '../engine/turn-context'

export interface CompactionTurnResult {
  readonly turn: CompletedTurn
  readonly compactionOutcome: { readonly isFallback: false; readonly compactResult: CompactResult } | { readonly isFallback: true }
  readonly inputTokens: number | null
  readonly outputTokens: number | null
}

export function runCompactionTurn(
  forkId: string | null,
  roleId: RoleId,
  windowState: ForkWindowState,
  softCap: number,
  publish: (event: AppEvent) => Effect.Effect<void>,
  read: any,
  agentStatus: AgentLifecycleState,
): Effect.Effect<CompactionTurnResult, any, any> {
  return Effect.gen(function* () {
    const agentDef = getAgentDefinition(roleId)

    // Resolve model (same as Cortex)
    const modelResolver = yield* AgentModelResolver
    const agentId = forkId
      ? getAgentByForkId(agentStatus, forkId)?.agentId ?? '000000000000'
      : '000000000000'
    const agentModel = yield* modelResolver.resolvePrimary(roleId, agentId)

    // Get toolkit and fork layer (same as Cortex)
    const ambientService = yield* AmbientServiceTag
    const configState = ambientService.getValue(ConfigAmbient)
    const sessionOptions = ambientService.getValue(SessionOptionsAmbient)
    const toolkit = getEffectiveToolkit(roleId, configState, undefined, { solo: sessionOptions.solo })
    const execManager = yield* ExecutionManager
    const forkLayer = execManager.getForkLayer(forkId)
    if (!forkLayer) {
      return yield* new CompactionTurnError({
        reason: 'ForkLayerMissing',
        message: 'Fork layer not initialized',
      })
    }

    // Session context
    const sessionCtx = yield* read(SessionContextProjection)
    const scratchpadPath = sessionCtx.context?.scratchpadPath ?? process.cwd()
    const skills = ambientService.getValue(SkillsAmbient)

    // Compute budget for compact() tool
    const { keptTailTokens } = computeCompactionSizing(windowState.messages, softCap)
    const sessionContextTokens = windowState.messages[0]?.estimatedTokens ?? 0
    const margin = 2000
    const maxPayloadTokens = Math.max(
      4000,
      softCap - windowState.systemPromptTokens - sessionContextTokens - keptTailTokens - margin,
    )

    // Create CompactionContextTag layer with shared result ref
    const compactResultRef = yield* Ref.make<CompactResult | null>(null)
    const compactionLayer = Layer.succeed(CompactionContextTag, {
      isCompacting: true as const,
      resultRef: compactResultRef,
      maxPayloadTokens,
    })

    const baseTurnLayer = Layer.merge(forkLayer, compactionLayer)

    // Retry loop: attempt up to COMPACTION_MAX_RETRIES times
    let lastTurn: CompletedTurn | null = null
    let lastInputTokens: number | null = null
    let lastOutputTokens: number | null = null

    for (let attempt = 0; attempt < COMPACTION_MAX_RETRIES; attempt++) {
      // Reset ref for each attempt
      yield* Ref.set(compactResultRef, null)

      const compactionTurnId = createId()
      const compactionChainId = createId()
      const turnContextLayer = Layer.succeed(TurnContextTag, {
        turnId: compactionTurnId,
        chainId: compactionChainId,
        forkId,
      })
      const operationLayer = Layer.succeed(AgentModelOperationContextTag, {
        operationKind: 'compact',
        operationId: compactionTurnId,
        chainId: compactionChainId,
        forkId,
      })
      const turnLayer = Layer.mergeAll(baseTurnLayer, turnContextLayer, operationLayer)
      const harness = createHarness({
        model: agentModel.model,
        toolkit,
        layer: turnLayer,
        hooks: buildStandardHooks({ forkId, turnId: compactionTurnId, agentDef, scratchpadPath }),
      })

      // Build system prompt (identical to normal turns → prefix cache preserved)
      const systemPrompt = buildSystemPrompt({
        roleDef: agentDef,
        skills,
        headless: sessionOptions.headless,
        systemPromptOverride: sessionOptions.systemPromptOverride,
      })

      // Build compaction prompt: full window + reflection instruction appended
      const timezone = sessionCtx.context?.timezone ?? null
      const formatter = createAgentFormatter(createToolResultFormatter(toolkit))
      const basePrompt = windowToPrompt({
        windowState,
        systemPrompt,
        timezone,
        formatter,
        autopilotEnabled: windowState.autopilotEnabled,
        leaderLastAutopilotKnowledge: windowState.consumerAutopilotKnowledge.leader,
      })
      const compactionPrompt = buildCompactionPrompt(basePrompt)

      const liveTurn = yield* Effect.provide(harness.runTurn(compactionPrompt), turnLayer)
      yield* Stream.runForEach(liveTurn.events, () => Effect.void)

      // Build CompletedTurn from canonical turn state
      const state = yield* Ref.get(liveTurn.state)
      const canonical = state.canonical
      const hasContent = (Option.isSome(canonical.assistantMessage.text) && canonical.assistantMessage.text.value.trim().length > 0)
        || (Option.isSome(canonical.assistantMessage.toolCalls) && canonical.assistantMessage.toolCalls.value.length > 0)

      if (hasContent) {
        lastTurn = {
          turnId: compactionTurnId,
          assistant: canonical.assistantMessage,
          toolResults: [...canonical.toolResults],
          feedback: [],
          clean: true,
        }
      }

      lastInputTokens = canonical.usage?.inputTokens ?? null
      lastOutputTokens = canonical.usage?.outputTokens ?? null

      // Check if compact() was called
      const compactResult = yield* Ref.get(compactResultRef)
      if (compactResult !== null) {
        return {
          turn: lastTurn!,
          compactionOutcome: { isFallback: false, compactResult },
          inputTokens: lastInputTokens,
          outputTokens: lastOutputTokens,
        }
      }

      if (attempt < COMPACTION_MAX_RETRIES - 1) {
        logger.warn({ forkId, attempt: attempt + 1 }, '[CompactionTurn] Agent did not call compact(), retrying')
      }
    }

    // All retries exhausted — fallback
    logger.warn({ forkId }, '[CompactionTurn] All compaction retries exhausted, falling back to tail preservation')

    if (!lastTurn) {
      return yield* new CompactionTurnError({
        reason: 'EmptyResponse',
        message: 'Empty compaction response after all retries',
      })
    }

    return {
      turn: lastTurn,
      compactionOutcome: { isFallback: true },
      inputTokens: lastInputTokens,
      outputTokens: lastOutputTokens,
    }
  })
}
