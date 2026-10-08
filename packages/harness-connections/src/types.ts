import type { HarnessSetup, HarnessSetupTarget } from "./setup"
import { Context, Data, Effect, Option, Schema } from "effect"
import type { ProviderModelId } from "@magnitudedev/sdk"
import { HarnessIdSchema, type HarnessId } from "@magnitudedev/sdk"

export { HarnessIdSchema, HARNESS_PRIORITY, type HarnessId } from "@magnitudedev/sdk"

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
  readonly operation: "list" | "connect" | "sync" | "disconnect" | "skill" | "startup" | "describe"
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
  /** The setup a person applies by hand on a computer that will run the harness against this Magnitude. */
  readonly describe: (target: Omit<HarnessSetupTarget, "model" | "models"> & { readonly model: ProviderModelId }) => Effect.Effect<HarnessSetup, HarnessConnectionError>
}
