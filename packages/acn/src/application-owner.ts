import { Context, Effect, Stream } from "effect"
import {
  ApplicationOwnerUnavailable,
  OwnerRequestFailed,
  OwnerRequestUnsupported,
  type ApplicationOwnerState,
  type OwnerRequest,
} from "@magnitudedev/acn-protocol"
import type { AcnOwnerControl } from "./owned-control"

/** The application that owns this ACN, reached over the private owned-control channel. */
export class AcnOwner extends Context.Tag("AcnOwner")<AcnOwner, AcnOwnerControl>() {}

export const watchApplicationOwner: Stream.Stream<ApplicationOwnerState, never, AcnOwner> =
  Stream.unwrap(Effect.map(AcnOwner, owner => owner.ownerState))

export const requestOwner = (request: OwnerRequest) => Effect.flatMap(AcnOwner, owner => owner.request(request)).pipe(
  Effect.mapError(error => new ApplicationOwnerUnavailable({ message: error.message })),
  Effect.flatMap((reply): Effect.Effect<{}, OwnerRequestUnsupported | OwnerRequestFailed> => {
    switch (reply._tag) {
      case "Done": return Effect.succeed({})
      case "Unsupported": return Effect.fail(new OwnerRequestUnsupported({ request: request._tag }))
      case "Failed": return Effect.fail(new OwnerRequestFailed({ message: reply.message }))
    }
  }),
)
