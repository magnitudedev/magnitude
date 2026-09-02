import { Context, Data, Effect, Option, Schema } from "effect"
import type { ProviderModelId } from "@magnitudedev/sdk"

export const HarnessIdSchema = Schema.Literal(
  "magnitude",
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
  "magnitude",
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
}

export interface HarnessLaunchPlan {
  readonly harness: HarnessId
  /** Ambient command name for user-facing handoff instructions. */
  readonly command: string
  /** Exact detected executable used when Magnitude launches the harness itself. */
  readonly executable: string
  readonly args: ReadonlyArray<string>
  readonly environment: Readonly<Record<string, string>>
  readonly modelId: ProviderModelId
}

export interface HarnessConnectOptions {
  /** Persist this model as the harness selection for ordinary new sessions. */
  readonly model: Option.Option<ProviderModelId>
}

export class HarnessConnectionError extends Data.TaggedError("HarnessConnectionError")<{
  readonly operation: "list" | "connect" | "sync" | "disconnect" | "launch" | "skill" | "startup"
  readonly harness?: HarnessId
  readonly message: string
}> {}

export interface HarnessConnection {
  readonly list: Effect.Effect<ReadonlyArray<HarnessDestination>, HarnessConnectionError>
  readonly connect: (
    harness: HarnessId,
    options: HarnessConnectOptions,
  ) => Effect.Effect<void, HarnessConnectionError>
  readonly launch: (
    harness: HarnessId,
    model: ProviderModelId,
  ) => Effect.Effect<HarnessLaunchPlan, HarnessConnectionError>
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

const magnitudeDestination: HarnessDestination = {
  id: HarnessIdSchema.make("magnitude"),
  name: "Magnitude Harness",
  availability: "Installed",
  selectable: true,
  connected: false,
  note: "Optimized for local models",
}

export const UnavailableHarnessConnection: HarnessConnection = {
  list: Effect.succeed([magnitudeDestination]),
  connect: (harness) => harness === "magnitude"
    ? Effect.void
    : Effect.fail(new HarnessConnectionError({ operation: "connect", harness, message: "External harness connections are unavailable in this client" })),
  launch: (harness, modelId) => harness === "magnitude"
    ? Effect.succeed({ harness, command: "magnitude", executable: "magnitude", args: [], environment: {}, modelId })
    : Effect.fail(new HarnessConnectionError({ operation: "launch", harness, message: "External harness connections are unavailable in this client" })),
  sync: () => Effect.succeed([magnitudeDestination]),
  disconnect: (harness) => Effect.fail(new HarnessConnectionError({ operation: "disconnect", harness, message: "Harness connections are unavailable in this client" })),
  installSkill: (harness) => Effect.fail(new HarnessConnectionError({ operation: "skill", harness, message: "Skill installation is unavailable in this client" })),
  installStartup: Effect.fail(new HarnessConnectionError({ operation: "startup", message: "Startup installation is unavailable in this client" })),
}
