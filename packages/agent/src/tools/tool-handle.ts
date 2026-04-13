import type { ToolCallEvent } from '@magnitudedev/xml-act'
import type { ToolStateEvent, StreamingAccumulatorLike } from '@magnitudedev/tools'
import { normalizeToolEvent } from '../normalizer'
import type { ToolState } from '../models'
import type { ToolKey, AgentCatalogEntry } from '../catalog'

export type { ToolState } from '../models'
type AnyToolEvent = ToolStateEvent<unknown, unknown, unknown>
type ToolReducer<S> = { bivarianceHack(state: S, event: AnyToolEvent): S }['bivarianceHack']

export interface ToolHandle {
  readonly toolKey: ToolKey
  readonly state: ToolState
  process(raw: ToolCallEvent): ToolHandle
  interrupt(): ToolHandle
}

type ToolStateFor<K extends ToolKey> = AgentCatalogEntry['state']['initial']

export function createToolHandle(toolKey: ToolKey, entry: AgentCatalogEntry): ToolHandle {
  const acc = entry.state.binding.createAccumulator()
  return buildHandle(toolKey, entry.state.initial, acc, entry.state.reduce as ToolReducer<ToolStateFor<typeof toolKey>>)
}

function buildHandle<K extends ToolKey>(
  toolKey: K,
  state: ToolStateFor<K>,
  acc: StreamingAccumulatorLike<unknown>,
  reduce: ToolReducer<ToolStateFor<K>>,
): ToolHandle {
  return {
    toolKey,
    get state() { return state },
    process(raw: ToolCallEvent): ToolHandle {
      acc.ingest(raw)
      const event = normalizeToolEvent(raw, acc.current)
      if (event) {
        const reduced = reduce(state, event)
        return buildHandle(toolKey, reduced, acc, reduce)
      }
      return this
    },
    interrupt(): ToolHandle {
      return buildHandle(toolKey, reduce(state, { type: 'interrupted' }), acc, reduce)
    },
  }
}
