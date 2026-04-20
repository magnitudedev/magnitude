import type { TurnEngineEvent, ToolLifecycleEvent } from '@magnitudedev/xml-act'
import type { ToolState } from '../models/tool-state'
import type { ToolKey, AgentCatalogEntry } from '../catalog'

export type { ToolState } from '../models/tool-state'
type ToolReducer<S> = { bivarianceHack(state: S, event: ToolLifecycleEvent): S }['bivarianceHack']

export interface ToolHandle {
  readonly toolKey: ToolKey
  readonly state: ToolState
  process(event: TurnEngineEvent): ToolHandle
  interrupt(): ToolHandle
}

type ToolStateFor<K extends ToolKey> = AgentCatalogEntry['state']['initial']

export function createToolHandle(toolKey: ToolKey, entry: AgentCatalogEntry): ToolHandle {
  return buildHandle(toolKey, entry.state.initial, entry.state.reduce as ToolReducer<ToolStateFor<typeof toolKey>>)
}

function isToolLifecycleEvent(event: TurnEngineEvent): event is ToolLifecycleEvent {
  switch (event._tag) {
    case 'ToolInputStarted':
    case 'ToolInputFieldChunk':
    case 'ToolInputFieldComplete':
    case 'ToolInputReady':
    case 'ToolInputParseError':
    case 'ToolExecutionStarted':
    case 'ToolExecutionEnded':
    case 'ToolEmission':
      return true
    default:
      return false
  }
}

function buildHandle<K extends ToolKey>(
  toolKey: K,
  state: ToolStateFor<K>,
  reduce: ToolReducer<ToolStateFor<K>>,
): ToolHandle {
  return {
    toolKey,
    get state() { return state },
    process(event: TurnEngineEvent): ToolHandle {
      if (!isToolLifecycleEvent(event)) return this
      const reduced = reduce(state, event)
      return buildHandle(toolKey, reduced, reduce)
    },
    interrupt(): ToolHandle {
      const interruptEvent: ToolLifecycleEvent = {
        _tag: 'ToolExecutionEnded',
        toolCallId: '',
        tagName: '',
        group: '',
        toolName: '',
        result: { _tag: 'Interrupted' },
      }
      return buildHandle(toolKey, reduce(state, interruptEvent), reduce)
    },
  }
}
