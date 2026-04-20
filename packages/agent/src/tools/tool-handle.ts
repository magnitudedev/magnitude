import type { RuntimeEvent } from '@magnitudedev/xml-act'
import { createParameterAccumulator, deriveParameters } from '@magnitudedev/xml-act'
import type { ToolStateEvent, StreamingAccumulatorLike } from '@magnitudedev/tools'
import { normalizeToolEvent } from '../normalizer'
import type { ToolState } from '../models/tool-state'
import type { ToolKey, AgentCatalogEntry } from '../catalog'

export type { ToolState } from '../models/tool-state'
type AnyToolEvent = ToolStateEvent<unknown, unknown, unknown>
type ToolReducer<S> = { bivarianceHack(state: S, event: AnyToolEvent): S }['bivarianceHack']

export interface ToolHandle {
  readonly toolKey: ToolKey
  readonly state: ToolState
  process(raw: RuntimeEvent): ToolHandle
  interrupt(): ToolHandle
}

type ToolStateFor<K extends ToolKey> = AgentCatalogEntry['state']['initial']

export function createToolHandle(toolKey: ToolKey, entry: AgentCatalogEntry): ToolHandle {
  // Derive parameter schema from the tool's input schema
  const toolSchema = deriveParameters(entry.tool.inputSchema.ast)
  // Create accumulator from the derived schema
  const acc = createParameterAccumulator(toolSchema, entry.tool.inputSchema.ast)
  return buildHandle(toolKey, entry.state.initial, acc, entry.state.reduce as ToolReducer<ToolStateFor<typeof toolKey>>)
}

function buildHandle<K extends ToolKey>(
  toolKey: K,
  state: ToolStateFor<K>,
  acc: StreamingAccumulatorLike<unknown, RuntimeEvent>,
  reduce: ToolReducer<ToolStateFor<K>>,
): ToolHandle {
  return {
    toolKey,
    get state() { return state },
    process(raw: RuntimeEvent): ToolHandle {
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
