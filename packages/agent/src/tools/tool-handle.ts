import type { TurnEngineEvent, ToolLifecycleEvent } from '@magnitudedev/turn-engine'
import type { ToolStateEvent } from '@magnitudedev/tools'
import type { ToolState } from '../models/tool-state'
import type { ToolKey, AgentCatalogEntry } from '../catalog'

export type { ToolState } from '../models/tool-state'
type ToolReducer<S> = { bivarianceHack(state: S, event: ToolStateEvent): S }['bivarianceHack']

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
    case 'ToolInputDecodeFailure':
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
      // Map xml-act ToolLifecycleEvent → ToolStateEvent (only ToolParseError differs)
      const mapped: ToolStateEvent = event._tag === 'ToolInputDecodeFailure'
        ? { _tag: 'ToolParseError', error: typeof event.detail === 'string' ? event.detail : String(event.detail) }
        : event as ToolStateEvent
      const reduced = reduce(state, mapped)
      return buildHandle(toolKey, reduced, reduce)
    },
    interrupt(): ToolHandle {
      const interruptEvent: ToolStateEvent = {
        _tag: 'ToolExecutionEnded',
        result: { _tag: 'Interrupted' },
      }
      return buildHandle(toolKey, reduce(state, interruptEvent), reduce)
    },
  }
}
