/**
 * liftTurnEngineEvent
 *
 * Pure function that maps a single TurnEngineEvent (turn-engine library vocabulary)
 * to zero or more AppEvents (agent bus vocabulary).
 *
 * Pipeline: response → turn-engine → app. This file is the boundary between
 * turn-engine and app event vocabularies.
 *
 * Mapping is 1:1 wherever possible. Tool lifecycle events (input streaming,
 * execution, emission, decode failures) all funnel through the unified
 * `tool_event { toolCallId, toolKey, event: TurnEngineEvent }` envelope.
 */

import type {
  TurnEngineEvent,
  RegisteredTool,
} from '@magnitudedev/turn-engine'
import type { AppEvent, MessageDestination } from './events'
import { isToolKey, type ToolKey } from './catalog'

export interface LiftEngineContext {
  readonly forkId: string | null
  readonly turnId: string
  readonly registeredTools: ReadonlyMap<string, RegisteredTool<unknown>>
  /**
   * toolCallId → ToolKey map, maintained by cortex across the turn. Some
   * engine tool events (ToolInputFieldChunk, ToolInputFieldComplete,
   * ToolInputReady, ToolEmission) carry only toolCallId; the toolKey for
   * those is discovered via this map populated when ToolInputStarted /
   * ToolExecutionStarted / ToolInputDecodeFailure events arrive (those carry
   * toolName).
   */
  readonly toolCallToToolKey: ReadonlyMap<string, ToolKey>
}

/** Resolve toolName → ToolKey via registered tools' meta.defKey. */
export function resolveToolKey(
  toolName: string,
  registeredTools: ReadonlyMap<string, RegisteredTool<unknown>>,
): ToolKey | null {
  const rt = registeredTools.get(toolName)
  if (!rt) return null
  const meta = rt.meta as { defKey?: unknown } | undefined
  const defKey = typeof meta?.defKey === 'string' ? meta.defKey : null
  if (!defKey) return null
  return isToolKey(defKey) ? (defKey as ToolKey) : null
}

/**
 * Lift a single TurnEngineEvent to zero or more AppEvents.
 *
 * Returns readonly AppEvent[] — publish in order. Empty for events that don't
 * surface to the app bus (e.g. TurnEnd; cortex builds turn_outcome from it).
 */
export function liftTurnEngineEvent(
  event: TurnEngineEvent,
  ctx: LiftEngineContext,
): readonly AppEvent[] {
  const { forkId, turnId, registeredTools } = ctx

  switch (event._tag) {
    // -------------------------------------------------------------------------
    // Thought
    // -------------------------------------------------------------------------
    case 'ThoughtStart':
      return [{ type: 'thinking_start', forkId, turnId }]

    case 'ThoughtChunk':
      return [{ type: 'thinking_chunk', forkId, turnId, text: event.text }]

    case 'ThoughtEnd':
      return [{ type: 'thinking_end', forkId, turnId }]

    // -------------------------------------------------------------------------
    // Message
    // -------------------------------------------------------------------------
    case 'MessageStart':
      return [{
        type: 'message_start',
        forkId,
        turnId,
        id: event.id,
        destination: mapDestination(event.to),
      }]

    case 'MessageChunk':
      return [{
        type: 'message_chunk',
        forkId,
        turnId,
        id: event.id,
        text: event.text,
      }]

    case 'MessageEnd':
      return [{
        type: 'message_end',
        forkId,
        turnId,
        id: event.id,
      }]

    // -------------------------------------------------------------------------
    // Tool lifecycle — all funnel through tool_event { event } envelope.
    // Includes input streaming, execution, emission, and input decode failures.
    //
    // toolKey resolution:
    //   - Events that carry toolName (ToolInputStarted, ToolExecutionStarted/Ended,
    //     ToolInputDecodeFailure): resolve via registered tools.
    //   - Events that carry only toolCallId (ToolInputFieldChunk/Complete,
    //     ToolInputReady, ToolEmission): look up via ctx.toolCallToToolKey,
    //     populated by cortex when prior events with toolName flowed through.
    // -------------------------------------------------------------------------
    case 'ToolInputStarted':
    case 'ToolExecutionStarted':
    case 'ToolExecutionEnded':
    case 'ToolInputDecodeFailure': {
      const toolKey = resolveToolKey(event.toolName, registeredTools)
      if (toolKey === null) return []
      return [{
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: event.toolCallId,
        toolKey,
        event,
      }]
    }
    case 'ToolInputFieldChunk':
    case 'ToolInputFieldComplete':
    case 'ToolInputReady':
    case 'ToolEmission': {
      const toolKey = ctx.toolCallToToolKey.get(event.toolCallId) ?? null
      if (toolKey === null) return []
      return [{
        type: 'tool_event',
        forkId,
        turnId,
        toolCallId: event.toolCallId,
        toolKey,
        event,
      }]
    }

    // -------------------------------------------------------------------------
    // Turn-level structural failures and turn end — handled by cortex via
    // TurnEnd.outcome / direct turn_outcome publication. Nothing on the bus here.
    // -------------------------------------------------------------------------
    case 'TurnStructureDecodeFailure':
    case 'TurnEnd':
      return []

    default: {
      const _exhaustive: never = event
      void _exhaustive
      return []
    }
  }
}

function mapDestination(to: string): MessageDestination {
  if (to === 'parent') return { kind: 'parent' }
  if (to.startsWith('worker:')) return { kind: 'worker', taskId: to.slice('worker:'.length) }
  return { kind: 'user' }
}
