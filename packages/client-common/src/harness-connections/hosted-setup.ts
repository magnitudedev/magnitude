import { Schema } from "effect"
import { ModelIdSchema } from "@magnitudedev/sdk"

/** Terminal hosting, independent of the daemon RPC protocol. */
export const HOSTED_SETUP_PROTOCOL_VERSION = 1
export const HOSTED_SETUP_MAX_RESULT_BYTES = 16 * 1024
export const HostedSetupCapability = Schema.Struct({
  protocolVersion: Schema.Literal(HOSTED_SETUP_PROTOCOL_VERSION),
})
const version = Schema.Literal(HOSTED_SETUP_PROTOCOL_VERSION)
export const HostedSetupResult = Schema.Union(
  Schema.TaggedStruct("Completed", { protocolVersion: version, modelId: ModelIdSchema }),
  Schema.TaggedStruct("Cancelled", { protocolVersion: version }),
  Schema.TaggedStruct("Failed", {
    protocolVersion: version,
    message: Schema.String.pipe(Schema.filter((message) => new TextEncoder().encode(message).length <= 4096)),
  }),
)
export type HostedSetupResult = typeof HostedSetupResult.Type

export const hostedSetupFailure = (message: string): HostedSetupResult => ({
  protocolVersion: HOSTED_SETUP_PROTOCOL_VERSION, _tag: "Failed",
  // Bound by UTF-8 bytes, including messages containing non-ASCII diagnostics.
  message: new TextDecoder().decode(new TextEncoder().encode(message).slice(0, 4093)),
})
