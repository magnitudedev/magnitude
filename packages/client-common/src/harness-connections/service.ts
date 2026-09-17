import { Context, Data, Effect, Option, Schema } from "effect"
import type { ProviderModelId } from "@magnitudedev/sdk"

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

const harnessPriorityValues = [
  "pi",
  "opencode",
  "hermes",
  "openclaw",
  "codex",
  "claude-code",
  "oh-my-pi",
  "cline",
] as const
export const HARNESS_PRIORITY: ReadonlyArray<HarnessId> = harnessPriorityValues.map(
  (value) => HarnessIdSchema.make(value),
)

export const HarnessAvailabilitySchema = Schema.Literal("Installed", "Not installed")
export type HarnessAvailability = typeof HarnessAvailabilitySchema.Type

export interface HarnessDestination {
  readonly id: HarnessId
  readonly name: string
  readonly availability: HarnessAvailability
  readonly selectable: boolean
  readonly connected: boolean
  readonly note?: string
  readonly companion?: HarnessCompanionDescription
  /** The harness connection is not useful without Magnitude's agent instructions. */
  readonly skillRequired?: boolean
}

export interface HarnessCompanionDescription {
  readonly name: string
  readonly source: string
  readonly securityNotice: string
}

export interface HarnessCompanionConnectionResult extends HarnessCompanionDescription {
  readonly status: "installed" | "enabled" | "already-installed"
  readonly activationInstructions: Option.Option<string>
}

export interface HarnessConnectResult {
  readonly companion: Option.Option<HarnessCompanionConnectionResult>
  readonly skillInstalled: boolean
  readonly startupInstalled: boolean
}

export interface HarnessConnectOptions {
  /** Persist this model as the harness selection for ordinary new sessions. */
  readonly model: Option.Option<ProviderModelId>
  /** Install or refresh the Magnitude skill when the connector does not require it. */
  readonly installSkill?: boolean
  /** Register the Magnitude desktop application for login startup. */
  readonly launchOnStartup?: boolean
}

export class HarnessConnectionError extends Data.TaggedError("HarnessConnectionError")<{
  readonly operation: "list" | "connect" | "sync" | "disconnect" | "skill" | "startup"
  readonly harness?: HarnessId
  readonly message: string
}> {}

export interface HarnessConnection {
  readonly list: Effect.Effect<ReadonlyArray<HarnessDestination>, HarnessConnectionError>
  readonly connect: (
    harness: HarnessId,
    options: HarnessConnectOptions,
  ) => Effect.Effect<HarnessConnectResult, HarnessConnectionError>
  readonly sync: (
    harness?: HarnessId,
  ) => Effect.Effect<ReadonlyArray<HarnessDestination>, HarnessConnectionError>
  readonly disconnect: (harness: HarnessId) => Effect.Effect<void, HarnessConnectionError>
  readonly installSkill: (harness: HarnessId) => Effect.Effect<void, HarnessConnectionError>
  readonly installStartup: Effect.Effect<void, HarnessConnectionError>
}

export const HarnessConnection = Context.GenericTag<HarnessConnection>(
  "client/HarnessConnection",
)

export const UnavailableHarnessConnection: HarnessConnection = {
  list: Effect.succeed([]),
  connect: (harness) => Effect.fail(new HarnessConnectionError({ operation: "connect", harness, message: "External harness connections are unavailable in this client" })),
  sync: (harness) => Effect.fail(new HarnessConnectionError({ operation: "sync", ...(harness === undefined ? {} : { harness }), message: "External harness connections are unavailable in this client" })),
  disconnect: (harness) => Effect.fail(new HarnessConnectionError({ operation: "disconnect", harness, message: "Harness connections are unavailable in this client" })),
  installSkill: (harness) => Effect.fail(new HarnessConnectionError({ operation: "skill", harness, message: "Skill installation is unavailable in this client" })),
  installStartup: Effect.fail(new HarnessConnectionError({ operation: "startup", message: "Startup installation is unavailable in this client" })),
}
