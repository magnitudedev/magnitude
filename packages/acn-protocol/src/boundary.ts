import type { Context } from "effect"
import * as Boundary from "@magnitudedev/effect-query/rpc"

/**
 * The ACN boundary. Every client↔ACN interaction is defined through it as a
 * query, a mutation, or a subscription; the wire group is assembled from the
 * same definitions' `rpc` values.
 */
export const Acn = Boundary.make("Acn")

/** The transport service definitions execute against (one per connection). */
export type AcnTransport = Context.Tag.Service<typeof Acn.Client>
