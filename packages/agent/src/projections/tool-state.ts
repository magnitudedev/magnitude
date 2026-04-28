/**
 * ToolStateProjection (Forked)
 *
 * Canonical owner of per-tool-call lifecycle state and model-backed parsed tool state.
 * This projection is intentionally display-agnostic: it tracks all tool calls,
 * including hidden tools, and exposes canonical handles keyed by tool call id.
 */

import { Projection } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'
import { catalog, type AgentCatalogEntry } from '../catalog'
import { createToolHandle, type ToolHandle } from '../tools/tool-handle'

export interface ToolStateProjectionState {
  readonly toolHandles: { readonly [callId: string]: ToolHandle }
}

export const ToolStateProjection = Projection.defineForked<AppEvent, ToolStateProjectionState>()({
  name: 'ToolState',

  initialFork: {
    toolHandles: {},
  },

  eventHandlers: {
    tool_event: ({ event, fork }) => {
      const inner = event.event

      if (inner._tag === 'ToolInputStarted') {
        if (!event.toolKey) return fork
        const entry = catalog.entries[event.toolKey] as AgentCatalogEntry | undefined
        if (!entry) return fork
        const handle = createToolHandle(event.toolKey, entry).process(inner)
        return {
          ...fork,
          toolHandles: {
            ...fork.toolHandles,
            [event.toolCallId]: handle,
          },
        }
      }

      const handle = fork.toolHandles[event.toolCallId]
      if (!handle) return fork

      const nextHandle = handle.process(inner)
      return {
        ...fork,
        toolHandles: {
          ...fork.toolHandles,
          [event.toolCallId]: nextHandle,
        },
      }
    },

    interrupt: ({ fork }) => {
      const nextToolHandles = Object.fromEntries(
        Object.entries(fork.toolHandles).map(([toolCallId, handle]) => [
          toolCallId,
          handle.interrupt(),
        ])
      )

      return {
        ...fork,
        toolHandles: nextToolHandles,
      }
    },
  },
})
