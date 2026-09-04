import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import type * as CommandExecutor from "@effect/platform/CommandExecutor"
import type {
  HarnessCompanionConnectionResult,
  HarnessCompanionDescription,
  HarnessId,
  HarnessLaunchPlan,
} from "@magnitudedev/client-common"
import {
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { Option, Schema, type Effect } from "effect"
import type { SkillInstallationTarget } from "./paths"

const CONNECTOR_MAX_OUTPUT_TOKENS = 32_768

const HarnessReasoningCapabilitiesSchema = Schema.Union(
  Schema.Struct({
    supported: Schema.Literal(true),
    efforts: Schema.Array(ReasoningEffortSchema).pipe(Schema.minItems(1)),
    defaultEffort: ReasoningEffortSchema,
  }),
  Schema.Struct({
    supported: Schema.Literal(false),
    efforts: Schema.Tuple(),
  }),
).pipe(Schema.filter((reasoning) => !reasoning.supported || (
  new Set(reasoning.efforts).size === reasoning.efforts.length
  && reasoning.efforts.includes(reasoning.defaultEffort)
), { message: () => "reasoning efforts must be unique and contain the default effort" }))

const HarnessModelCapabilitiesSchema = Schema.Struct({
  vision: Schema.Boolean,
  tools: Schema.Boolean,
  structuredOutput: Schema.Boolean,
  reasoning: HarnessReasoningCapabilitiesSchema,
})

const ContextWindowSchema = Schema.Number.pipe(
  Schema.int(),
  Schema.positive(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)

const HarnessModelFields = {
  id: ProviderModelIdSchema,
  name: Schema.NonEmptyString,
  description: Schema.String,
  contextWindow: ContextWindowSchema,
  capabilities: HarnessModelCapabilitiesSchema,
} as const

export const HarnessModelSchema = Schema.transform(
  Schema.Struct({
    ...HarnessModelFields,
    maxOutputTokens: Schema.optionalWith(ContextWindowSchema, { as: "Option", exact: true }),
  }),
  Schema.typeSchema(Schema.Struct({ ...HarnessModelFields, maxOutputTokens: ContextWindowSchema })),
  {
    strict: true,
    decode: (model) => ({
      ...model,
      maxOutputTokens: Option.getOrElse(
        model.maxOutputTokens,
        () => Math.min(model.contextWindow, CONNECTOR_MAX_OUTPUT_TOKENS),
      ),
    }),
    encode: (model) => ({ ...model, maxOutputTokens: Option.some(model.maxOutputTokens) }),
  },
)
export type HarnessModel = typeof HarnessModelSchema.Type

export const HarnessRestoreSchema = Schema.Struct({
  model: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export type HarnessRestore = typeof HarnessRestoreSchema.Type

export const HarnessCompanionStateSchema = Schema.Struct({
  identity: Schema.NonEmptyString,
  source: Schema.NonEmptyString,
  ownership: Schema.Literal("magnitude", "pre-existing"),
  previousEntryJson: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})
export type HarnessCompanionState = typeof HarnessCompanionStateSchema.Type

export interface HarnessCompanionReconcileSpec {
  readonly installation: HarnessInstallation
  readonly previous: Option.Option<HarnessCompanionState>
}

export interface HarnessCompanionDisconnectSpec {
  readonly installation: HarnessInstallation
  readonly state: HarnessCompanionState
}

export interface HarnessCompanionPackage {
  readonly description: HarnessCompanionDescription
  readonly activation: HarnessCompanionConnectionResult["activation"]
  readonly reconcile: (
    spec: HarnessCompanionReconcileSpec,
  ) => Effect.Effect<{
    readonly state: HarnessCompanionState
    readonly status: HarnessCompanionConnectionResult["status"]
    /** Reverses non-file package-manager side effects when the enclosing connection transaction fails. */
    readonly rollback: Effect.Effect<
      void,
      unknown,
      FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
    >
  }, unknown, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor>
  readonly disconnect: (
    spec: HarnessCompanionDisconnectSpec,
  ) => Effect.Effect<{
    /** Reverses non-file package-manager effects when the enclosing disconnect transaction fails. */
    readonly rollback: Effect.Effect<
      void,
      unknown,
      FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
    >
  }, unknown, FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor>
}

export interface HarnessConnectionSpec {
  readonly models: ReadonlyArray<HarnessModel>
  readonly installation: HarnessInstallation
  /** Model to persist for ordinary new harness sessions. */
  readonly model: Option.Option<ProviderModelId>
  /** Last connector-owned projection, used to distinguish safe sync updates from user overrides. */
  readonly previousModels?: ReadonlyArray<HarnessModel>
}

export interface HarnessDisconnectionSpec {
  readonly models: ReadonlyArray<HarnessModel>
  readonly restore: Option.Option<HarnessRestore>
}

export interface HarnessInstallation {
  readonly executable: string
}

export interface HarnessConnector {
  readonly id: HarnessId
  readonly name: string
  /** Ambient command name; detection resolves it to an exact installed executable. */
  readonly executable: string
  readonly recommended?: boolean
  readonly note?: string
  readonly requiresStartup?: boolean
  readonly companion?: HarnessCompanionPackage
  readonly skillInstallationTarget: SkillInstallationTarget
  readonly skillRequired?: boolean
  readonly configurationFiles: ReadonlyArray<string>
  readonly detect: (searchPath: string) => Effect.Effect<Option.Option<HarnessInstallation>>
  readonly connect: (
    spec: HarnessConnectionSpec,
  ) => Effect.Effect<
    Option.Option<HarnessRestore>,
    unknown,
    FileSystem.FileSystem | Path.Path | CommandExecutor.CommandExecutor
  >
  readonly disconnect: (
    spec: HarnessDisconnectionSpec,
  ) => Effect.Effect<void, unknown, FileSystem.FileSystem | Path.Path>
  readonly launch: (
    modelId: ProviderModelId,
    installation: HarnessInstallation,
  ) => HarnessLaunchPlan
}
