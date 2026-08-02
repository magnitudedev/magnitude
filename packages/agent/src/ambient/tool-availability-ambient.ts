import { Ambient } from '@magnitudedev/event-core'
import { Effect, Schema } from 'effect'

export const WebSearchAvailabilitySchema = Schema.Union(
  Schema.TaggedStruct('Available', {
    source: Schema.Literal('magnitude', 'exa'),
  }),
  Schema.TaggedStruct('Unavailable', {}),
)
export type WebSearchAvailability = typeof WebSearchAvailabilitySchema.Type

export const ToolAvailabilityStateSchema = Schema.Struct({
  webSearch: WebSearchAvailabilitySchema,
})
export type ToolAvailabilityState = typeof ToolAvailabilityStateSchema.Type

const toolAvailabilityStateEquivalent = Schema.equivalence(ToolAvailabilityStateSchema)

export const sameToolAvailabilityState = (
  left: ToolAvailabilityState,
  right: ToolAvailabilityState,
): boolean => toolAvailabilityStateEquivalent(left, right)

export const INITIAL_TOOL_AVAILABILITY: ToolAvailabilityState = {
  webSearch: { _tag: 'Unavailable' },
}

export const ToolAvailabilityAmbient = Ambient.define<ToolAvailabilityState, never>({
  name: 'ToolAvailability',
  initial: Effect.succeed(INITIAL_TOOL_AVAILABILITY),
})
