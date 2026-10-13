import { Schema } from "effect"
import { MagnitudeHealthResponseSchema } from "./schemas/acn-health"
import { ApplicationOwnerState, OwnerReply, OwnerRequest } from "./schemas/application-owner"

/** Private inherited channel; never a client model API or a discoverable listener. */
export const DesktopOwnerCommand = Schema.Union(
  Schema.TaggedStruct("Start", {}),
  Schema.TaggedStruct("Shutdown", {}),
  Schema.TaggedStruct("StoppingObserved", {}),
  Schema.TaggedStruct("OwnerState", { state: ApplicationOwnerState }),
  Schema.TaggedStruct("OwnerResponse", { id: Schema.Int.pipe(Schema.nonNegative()), reply: OwnerReply }),
)
export type DesktopOwnerCommand = typeof DesktopOwnerCommand.Type

export const DesktopChildEvent = Schema.Union(
  Schema.TaggedStruct("Booted", {
    pid: Schema.Int.pipe(Schema.positive()),
  }),
  Schema.TaggedStruct("Health", { health: MagnitudeHealthResponseSchema }),
  Schema.TaggedStruct("OwnerRequest", { id: Schema.Int.pipe(Schema.nonNegative()), request: OwnerRequest }),
)
export type DesktopChildEvent = typeof DesktopChildEvent.Type
