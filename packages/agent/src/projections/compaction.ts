/**
 * CompactionProjection (Forked)
 *
 * Owns compaction-related state per fork as an FSM with ambient fields.
 */

import { Data } from 'effect'
import { Projection, Signal, FSM } from '@magnitudedev/event-core'
const { defineFSM } = FSM

import type { AppEvent, SessionContext } from '../events'
import { AgentRoutingProjection } from './agent-routing'
import { AgentStatusProjection, type AgentStatusState } from './agent-status'
import { UserMessageResolutionProjection } from './user-message-resolution'
import { CanonicalTurnProjection } from './canonical-turn'
import { ConfigAmbient, getSlotConfig, type SlotConfig, type ConfigState } from '../ambient/config-ambient'
import { SkillsAmbient } from '../ambient/skills-ambient'
import type { Skill } from '@magnitudedev/skills'

import { CHARS_PER_TOKEN } from '../constants'

import type { AgentVariant } from '../agents/variants'
import { getAgentDefinition, getAgentSlot, getForkInfo } from '../agents/registry'
import { buildSessionContextContent } from '../prompts/session-context'
import { renderSystemPrompt } from '../prompts/system-prompt'
import { buildResolvedToolSet } from '../tools/resolved-toolset'
import { ContentPart } from '../content'

// =============================================================================
// Context Limit Helpers
// =============================================================================

function isCompactionBlocking(tag: CompactionState['_tag']): boolean {
  return tag !== 'idle'
}

function deriveShouldCompact(
  tag: CompactionState['_tag'],
  tokenEstimate: number,
  limits: SlotConfig
): boolean {
  return tag === 'idle' && tokenEstimate > limits.softCap
}

/** Compute whether turns should be blocked due to context limit */
function computeContextLimitBlocked(
  tag: CompactionState['_tag'],
  tokenEstimate: number,
  limits: SlotConfig
): boolean {
  return isCompactionBlocking(tag) && tokenEstimate >= limits.hardCap
}

/** Estimate tokens for content string or content parts */
function estimateContentTokens(content: string): number
function estimateContentTokens(content: ContentPart[], modelId?: string | null, providerId?: string | null): number
function estimateContentTokens(content: string | ContentPart[], modelId?: string | null, providerId?: string | null): number {
  if (typeof content === 'string') {
    return Math.ceil(content.length / CHARS_PER_TOKEN)
  }
  let tokens = 0
  for (const part of content) {
    switch (part.type) {
      case 'text':
        tokens += Math.ceil(part.text.length / CHARS_PER_TOKEN)
        break
      case 'image':
        tokens += getImageTokenEstimator(modelId ?? null, providerId ?? null)(part.width, part.height)
        break
    }
  }
  return tokens
}

const systemPromptTokenCache = new Map<string, number>()

function estimateSystemPromptTokens(variant: AgentVariant, skills: Map<string, Skill>, configState: ConfigState): number {
  const cacheKey = variant
  const cached = systemPromptTokenCache.get(cacheKey)
  if (cached !== undefined) return cached
  
  const agentDef = getAgentDefinition(variant)
  const slot = getAgentSlot(variant)
  const toolSet = buildResolvedToolSet(agentDef, configState, slot)
  const prompt = renderSystemPrompt(agentDef, skills, toolSet)
  
  const tokens = Math.ceil(prompt.length / CHARS_PER_TOKEN)
  systemPromptTokenCache.set(cacheKey, tokens)
  return tokens
}

type ImageTokenEstimator = (width: number, height: number) => number

const estimators: Record<string, ImageTokenEstimator> = {
  anthropic: (w, h) => {
    const longEdge = Math.max(w, h)
    if (longEdge > 1568) {
      const scale = 1568 / longEdge
      w = Math.round(w * scale)
      h = Math.round(h * scale)
    }
    return Math.ceil((w * h) / 750)
  },

  openai: (w, h) => {
    const maxDim = Math.max(w, h)
    if (maxDim > 2048) {
      const scale = 2048 / maxDim
      w = Math.round(w * scale)
      h = Math.round(h * scale)
    }
    const minDim = Math.min(w, h)
    if (minDim > 768) {
      const scale = 768 / minDim
      w = Math.round(w * scale)
      h = Math.round(h * scale)
    }
    const tilesW = Math.ceil(w / 512)
    const tilesH = Math.ceil(h / 512)
    return (tilesW * tilesH * 170) + 85
  },

  google: () => 560,
  'google-ai': () => 560,
  'vertex-ai': () => 560,
}

function getImageTokenEstimator(modelId: string | null, providerId: string | null): ImageTokenEstimator {
  if (modelId) {
    if (/claude/i.test(modelId)) return estimators.anthropic
    if (/gpt|o[1-9]-/i.test(modelId)) return estimators.openai
    if (/gemini/i.test(modelId)) return estimators.google
  }

  if (providerId === 'anthropic' || providerId === 'aws-bedrock') return estimators.anthropic
  if (providerId === 'openai') return estimators.openai
  if (providerId === 'google' || providerId === 'google-ai' || providerId === 'vertex-ai') return estimators.google

  return estimators.anthropic
}

function asAgentVariant(role: string): AgentVariant | null {
  if (
    role === 'lead'
    || role === 'lead-oneshot'
    || role === 'worker'
  ) {
    return role
  }
  return null
}

// =============================================================================
// FSM State
// =============================================================================

interface AmbientCompactionFields {
  readonly tokenEstimate: number
  readonly lastActualInputTokens: number | null
  readonly hasCompletedTurn: boolean
  readonly modelId: string | null
  readonly providerId: string | null
  readonly contextLimitBlocked: boolean
  readonly shouldCompact: boolean
}

export class CompactionIdle extends Data.TaggedClass('idle')<AmbientCompactionFields> {}

export class Compacting extends Data.TaggedClass('compacting')<AmbientCompactionFields & {
  readonly compactedMessageCount: number
}> {}

export class PendingFinalization extends Data.TaggedClass('pendingFinalization')<AmbientCompactionFields & {
  readonly summary: string
  readonly compactedMessageCount: number
  readonly originalTokenEstimate: number
  readonly refreshedContext: SessionContext | null
}> {}

export const CompactionLifecycle = defineFSM(
  { idle: CompactionIdle, compacting: Compacting, pendingFinalization: PendingFinalization },
  { idle: ['compacting'], compacting: ['pendingFinalization', 'idle'], pendingFinalization: ['idle'] }
)

export type CompactionState =
  | CompactionIdle
  | Compacting
  | PendingFinalization

function emitLifecycleSignals(
  oldState: CompactionState,
  newState: CompactionState,
  forkId: string | null,
  emit: {
    readonly shouldCompactChanged: (value: { forkId: string | null; shouldCompact: boolean }) => void
    readonly compactionBlockingChanged: (value: { forkId: string | null; blocking: boolean }) => void
    readonly contextLimitBlockedChanged: (value: { forkId: string | null; blocked: boolean }) => void
  }
): void {
  if (oldState.shouldCompact !== newState.shouldCompact) {
    emit.shouldCompactChanged({ forkId, shouldCompact: newState.shouldCompact })
  }

  const oldBlocking = isCompactionBlocking(oldState._tag)
  const newBlocking = isCompactionBlocking(newState._tag)
  if (oldBlocking !== newBlocking) {
    emit.compactionBlockingChanged({ forkId, blocking: newBlocking })
  }

  if (oldState.contextLimitBlocked !== newState.contextLimitBlocked) {
    emit.contextLimitBlockedChanged({ forkId, blocked: newState.contextLimitBlocked })
  }
}

function withAmbient(
  state: CompactionState,
  updates: Partial<AmbientCompactionFields>
): CompactionState {
  return CompactionLifecycle.hold(state, updates)
}

function getForkConfig(
  configState: ConfigState,
  agentStatus: AgentStatusState,
  forkId: string | null,
): SlotConfig | null {
  const info = getForkInfo(agentStatus, forkId)
  if (!info) return null
  const slot = info.slot
  if (!slot) return null
  return getSlotConfig(configState, slot)
}

function recomputeOperationalFields(
  fork: CompactionState,
  limits: SlotConfig,
  updates: Partial<AmbientCompactionFields> = {}
): CompactionState {
  const tokenEstimate = updates.tokenEstimate ?? fork.tokenEstimate
  return withAmbient(fork, {
    ...updates,
    shouldCompact: deriveShouldCompact(fork._tag, tokenEstimate, limits),
    contextLimitBlocked: computeContextLimitBlocked(fork._tag, tokenEstimate, limits),
  })
}

// =============================================================================
// Projection
// =============================================================================

export const CompactionProjection = Projection.defineForked<AppEvent, CompactionState>()({
  name: 'Compaction',

  reads: [AgentRoutingProjection, AgentStatusProjection, UserMessageResolutionProjection, CanonicalTurnProjection] as const,
  ambients: [ConfigAmbient, SkillsAmbient] as const,

  signals: {
    shouldCompactChanged: Signal.create<{ forkId: string | null; shouldCompact: boolean }>('Compaction/shouldCompactChanged'),
    compactionBlockingChanged: Signal.create<{ forkId: string | null; blocking: boolean }>('Compaction/compactionBlockingChanged'),
    contextLimitBlockedChanged: Signal.create<{ forkId: string | null; blocked: boolean }>('Compaction/contextLimitBlockedChanged'),
  },

  initialFork: new CompactionIdle({
    tokenEstimate: 0,
    lastActualInputTokens: null,
    hasCompletedTurn: false,
    modelId: null,
    providerId: null,
    contextLimitBlocked: false,
    shouldCompact: false,
  }),

  eventHandlers: {
    session_initialized: ({ event, fork, emit, ambient, read }) => {
      const content = buildSessionContextContent(event.context)
      const contentTokens = estimateContentTokens(content)
      const skills = ambient.get(SkillsAmbient)
      const configState = ambient.get(ConfigAmbient)
      const tokenEstimate = estimateSystemPromptTokens('lead', skills, configState) + contentTokens
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, event.forkId)
      if (!limits) return fork
      const nextState = recomputeOperationalFields(fork, limits, {
        tokenEstimate,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    turn_completed: ({ event, fork, emit, read, ambient }) => {
      const canonical = read(CanonicalTurnProjection)
      const completedText = canonical.lastCompleted?.turnId === event.turnId
        ? canonical.lastCompleted.canonicalXml
        : ''
      const addedTokens = estimateContentTokens(completedText)
      const tokenEstimate = event.inputTokens !== null
        ? event.inputTokens + addedTokens
        : fork.tokenEstimate + addedTokens

      const configState = ambient.get(ConfigAmbient)
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, event.forkId)
      if (!limits) return fork
      const nextState = recomputeOperationalFields(fork, limits, {
        tokenEstimate,
        lastActualInputTokens: event.inputTokens ?? fork.lastActualInputTokens,
        hasCompletedTurn: true,
        modelId: event.modelId ?? fork.modelId,
        providerId: event.providerId ?? fork.providerId,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    turn_unexpected_error: ({ event, fork, emit, ambient, read }) => {
      const tokenEstimate = fork.tokenEstimate + estimateContentTokens(event.message)
      const configState = ambient.get(ConfigAmbient)
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, event.forkId)
      if (!limits) return fork
      const nextState = recomputeOperationalFields(fork, limits, {
        tokenEstimate,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    compaction_started: ({ event, fork, emit, ambient, read }) => {
      if (fork._tag !== 'idle') return fork

      const configState = ambient.get(ConfigAmbient)
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, event.forkId)
      if (!limits) return fork
      const nextState = CompactionLifecycle.transition(fork, 'compacting', {
        compactedMessageCount: event.compactedMessageCount,
        shouldCompact: false,
        contextLimitBlocked: computeContextLimitBlocked('compacting', fork.tokenEstimate, limits),
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    compaction_ready: ({ event, fork, emit, ambient, read }) => {
      if (fork._tag !== 'compacting') return fork

      const configState = ambient.get(ConfigAmbient)
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, event.forkId)
      if (!limits) return fork
      const nextState = CompactionLifecycle.transition(fork, 'pendingFinalization', {
        summary: event.summary,
        compactedMessageCount: event.compactedMessageCount,
        originalTokenEstimate: event.originalTokenEstimate,
        refreshedContext: event.refreshedContext,
        shouldCompact: false,
        contextLimitBlocked: computeContextLimitBlocked('pendingFinalization', fork.tokenEstimate, limits),
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    compaction_completed: ({ event, fork, emit, ambient, read }) => {
      if (fork._tag !== 'pendingFinalization') return fork

      const tokenEstimate = Math.max(0, fork.tokenEstimate - event.tokensSaved)
      const configState = ambient.get(ConfigAmbient)
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, event.forkId)
      if (!limits) return fork
      const nextState = CompactionLifecycle.transition(fork, 'idle', {
        tokenEstimate,
        shouldCompact: deriveShouldCompact('idle', tokenEstimate, limits),
        contextLimitBlocked: false,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    compaction_failed: ({ event, fork, emit }) => {
      if (fork._tag === 'idle') {
        if (!fork.contextLimitBlocked) return fork
        const nextState = withAmbient(fork, {
          contextLimitBlocked: false,
          shouldCompact: false,
        })
        emitLifecycleSignals(fork, nextState, event.forkId, emit)
        return nextState
      }

      const nextState = CompactionLifecycle.transition(fork, 'idle', {
        shouldCompact: false,
        contextLimitBlocked: false,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    interrupt: ({ event, fork, emit }) => {
      if (fork._tag === 'idle') {
        if (!fork.contextLimitBlocked) return fork
        const nextState = withAmbient(fork, {
          contextLimitBlocked: false,
          shouldCompact: false,
        })
        emitLifecycleSignals(fork, nextState, event.forkId, emit)
        return nextState
      }

      const nextState = CompactionLifecycle.transition(fork, 'idle', {
        shouldCompact: false,
        contextLimitBlocked: false,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },

    context_limit_hit: ({ event, fork, emit }) => {
      const nextState = withAmbient(fork, {
        contextLimitBlocked: true,
        shouldCompact: fork._tag === 'idle' ? true : fork.shouldCompact,
      })

      emitLifecycleSignals(fork, nextState, event.forkId, emit)
      return nextState
    },
  },

  signalHandlers: (on) => [
    on(AgentRoutingProjection.signals.agentRegistered, ({ value, state, ambient, read }) => {
      const { forkId, parentForkId, role } = value
      const parentState = state.forks.get(parentForkId)
      if (!parentState) {
        throw new Error(`Parent fork ${parentForkId} not found in CompactionProjection`)
      }

      const variant = asAgentVariant(role)
      if (variant === null) {
        throw new Error(`Unknown agent variant: ${role}`)
      }

      const skills = ambient.get(SkillsAmbient)
      const configState = ambient.get(ConfigAmbient)
      const tokenEstimate = estimateSystemPromptTokens(variant, skills, configState)
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, forkId)
      const newForkState = new CompactionIdle({
        tokenEstimate,
        lastActualInputTokens: null,
        hasCompletedTurn: false,
        modelId: parentState.modelId,
        providerId: parentState.providerId,
        shouldCompact: limits ? deriveShouldCompact('idle', tokenEstimate, limits) : false,
        contextLimitBlocked: false,
      })

      return {
        ...state,
        forks: new Map(state.forks).set(forkId, newForkState),
      }
    }),

    on(UserMessageResolutionProjection.signals.userMessageResolved, ({ value, state, emit, ambient, read }) => {
      const fork = state.forks.get(value.forkId)
      if (!fork) return state

      const addedTokens = estimateContentTokens([...value.content], fork.modelId, fork.providerId)
      const tokenEstimate = fork.tokenEstimate + addedTokens
      const configState = ambient.get(ConfigAmbient)
      const agentStatus = read(AgentStatusProjection)
      const limits = getForkConfig(configState, agentStatus, value.forkId)
      if (!limits) return state
      const nextState = recomputeOperationalFields(fork, limits, {
        tokenEstimate,
      })

      emitLifecycleSignals(fork, nextState, value.forkId, emit)

      return {
        ...state,
        forks: new Map(state.forks).set(value.forkId, nextState),
      }
    }),
  ],

  ambientHandlers: (on) => ([
    on(SkillsAmbient, ({ state }) => state),
    on(ConfigAmbient, ({ value, state, emit, read }) => {
      const nextForks = new Map<string | null, CompactionState>()
      const agentStatus = read(AgentStatusProjection)

      for (const [forkId, fork] of state.forks) {
        const limits = getForkConfig(value, agentStatus, forkId)
        if (!limits) {
          nextForks.set(forkId, fork)
          continue
        }
        const nextFork = recomputeOperationalFields(fork, limits)
        emitLifecycleSignals(fork, nextFork, forkId, emit)
        nextForks.set(forkId, nextFork)
      }

      return {
        ...state,
        forks: nextForks,
      }
    }),
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ] as any),
})