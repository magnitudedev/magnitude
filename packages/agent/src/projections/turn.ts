/**
 * TurnProjection (Forked)
 *
 * Turn lifecycle tracking for debugging and stats, per-fork.
 * Each fork has independent turn state to prevent cross-fork contamination.
 */

import { Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import type { ToolKey } from '../tools/tool-definitions'
import type { XmlToolResult } from '@magnitudedev/xml-act'

// =============================================================================
// Types
// =============================================================================

export interface ToolCall {
  readonly toolCallId: string
  readonly toolKey: ToolKey
  readonly input: unknown
  readonly result?: XmlToolResult
}

export interface TurnState {
  readonly activeTurn: {
    readonly turnId: string
    readonly chainId: string
    readonly toolCalls: readonly ToolCall[]
  } | null
  readonly completedTurns: number
}

// =============================================================================
// Projection
// =============================================================================

export const TurnProjection = Projection.defineForked<AppEvent, TurnState>()({
  name: 'Turn',

  initialFork: {
    activeTurn: null,
    completedTurns: 0
  },

  eventHandlers: {
    turn_started: ({ event, fork }) => ({
      ...fork,
      activeTurn: {
        turnId: event.turnId,
        chainId: event.chainId,
        toolCalls: []
      }
    }),

    tool_event: ({ event, fork }) => {
      if (!fork.activeTurn || fork.activeTurn.turnId !== event.turnId) {
        return fork
      }

      switch (event.event._tag) {
        case 'ToolInputStarted':
          return {
            ...fork,
            activeTurn: {
              ...fork.activeTurn,
              toolCalls: [
                ...fork.activeTurn.toolCalls,
                {
                  toolCallId: event.toolCallId,
                  toolKey: event.toolKey,
                  input: undefined,
                }
              ]
            }
          }

        case 'ToolInputReady': {
          const inner = event.event
          return {
            ...fork,
            activeTurn: {
              ...fork.activeTurn,
              toolCalls: fork.activeTurn.toolCalls.map(tc =>
                tc.toolCallId === event.toolCallId
                  ? { ...tc, input: inner.input }
                  : tc
              )
            }
          }
        }

        case 'ToolExecutionEnded': {
          const inner = event.event
          return {
            ...fork,
            activeTurn: {
              ...fork.activeTurn,
              toolCalls: fork.activeTurn.toolCalls.map(tc =>
                tc.toolCallId === event.toolCallId
                  ? { ...tc, result: inner.result }
                  : tc
              )
            }
          }
        }

        default:
          return fork
      }
    },

    turn_completed: ({ fork }) => ({
      ...fork,
      activeTurn: null,
      completedTurns: fork.completedTurns + 1
    }),

    turn_unexpected_error: ({ fork }) => ({
      ...fork,
      activeTurn: null
    }),

    interrupt: ({ fork }) => ({
      ...fork,
      activeTurn: null
    }),

  }
})
