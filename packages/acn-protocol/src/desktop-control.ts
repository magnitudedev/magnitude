import { Schema } from "effect"
import { MagnitudeHealthResponseSchema } from "./schemas/acn-health"

/** Private inherited channel; never a client model API or a discoverable listener. */
export const DesktopOwnerCommand = Schema.Union(
  Schema.TaggedStruct("Start", {}),
  Schema.TaggedStruct("Shutdown", {}),
  Schema.TaggedStruct("StoppingObserved", {}),
)
export type DesktopOwnerCommand = typeof DesktopOwnerCommand.Type

export const DesktopChildEvent = Schema.Union(
  Schema.TaggedStruct("Booted", {
    pid: Schema.Int.pipe(Schema.positive()),
  }),
  Schema.TaggedStruct("Health", { health: MagnitudeHealthResponseSchema }),
)
export type DesktopChildEvent = typeof DesktopChildEvent.Type
