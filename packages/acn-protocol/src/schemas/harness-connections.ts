import { Schema } from "effect"
import { ProviderModelIdSchema } from "@magnitudedev/ai/provider/model"

export const HarnessIdSchema = Schema.Literal(
  "pi",
  "opencode",
  "hermes",
  "openclaw",
  "codex",
  "claude-code",
  "oh-my-pi",
  "cline",
).pipe(Schema.brand("HarnessId"))
export type HarnessId = typeof HarnessIdSchema.Type

export const HARNESS_PRIORITY: ReadonlyArray<HarnessId> = HarnessIdSchema.from.literals.map(
  (value) => HarnessIdSchema.make(value),
)

/** Whether a harness's Magnitude configuration is in place on the machine running Magnitude. */
export const ConnectionInspection = Schema.Union(
  Schema.TaggedStruct("Connected", {}),
  Schema.TaggedStruct("Disconnected", { reason: Schema.String }),
  Schema.TaggedStruct("Unavailable", { reason: Schema.String }),
)
export type ConnectionInspection = typeof ConnectionInspection.Type

export const HarnessConnectionStatus = Schema.Struct({
  id: HarnessIdSchema,
  name: Schema.String,
  installed: Schema.Boolean,
  managed: Schema.Boolean,
  inspection: ConnectionInspection,
  configurationFiles: Schema.Array(Schema.String),
  plugin: Schema.optionalWith(Schema.Struct({ name: Schema.String, source: Schema.String }), { as: "Option", exact: true }),
})
export type HarnessConnectionStatus = typeof HarnessConnectionStatus.Type

export const HarnessConnectionsSnapshot = Schema.Union(
  Schema.TaggedStruct("Ready", { connections: Schema.Array(HarnessConnectionStatus) }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
)
export type HarnessConnectionsSnapshot = typeof HarnessConnectionsSnapshot.Type

export const HarnessConnectRequest = Schema.Struct({
  harness: HarnessIdSchema,
  model: Schema.optionalWith(ProviderModelIdSchema, { as: "Option", exact: true }),
  installSkill: Schema.Boolean,
})
export type HarnessConnectRequest = typeof HarnessConnectRequest.Type

export const HarnessCompanionStatus = Schema.Literal("installed", "enabled", "already-installed")
export const HarnessConnectOutcome = Schema.Struct({
  companion: Schema.optionalWith(Schema.Struct({
    name: Schema.String,
    source: Schema.String,
    securityNotice: Schema.String,
    status: HarnessCompanionStatus,
    activationInstructions: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }), { as: "Option", exact: true }),
  skillInstalled: Schema.Boolean,
})
export type HarnessConnectOutcome = typeof HarnessConnectOutcome.Type

export const HarnessConnectionOperation = Schema.Literal("list", "connect", "sync", "disconnect", "skill", "startup")
export class HarnessConnectionFailed extends Schema.TaggedError<HarnessConnectionFailed>()("HarnessConnectionFailed", {
  operation: HarnessConnectionOperation,
  harness: Schema.optionalWith(HarnessIdSchema, { as: "Option", exact: true }),
  message: Schema.String,
}) {}
