import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import type {
  HarnessId,
  HarnessLaunchPlan,
} from "@magnitudedev/client-common"
import {
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type ProviderModelId,
} from "@magnitudedev/sdk"
import { Schema, type Effect, type Option } from "effect"
import type { SkillInstallationTarget } from "./paths"

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

export const HarnessModelSchema = Schema.Struct({
  id: ProviderModelIdSchema,
  name: Schema.NonEmptyString,
  description: Schema.String,
  contextWindow: Schema.Number.pipe(
    Schema.int(),
    Schema.positive(),
    Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
  ),
  capabilities: HarnessModelCapabilitiesSchema,
})
export type HarnessModel = typeof HarnessModelSchema.Type

export interface HarnessConnectionSpec {
  readonly models: ReadonlyArray<HarnessModel>
  readonly setCurrent: Option.Option<ProviderModelId>
}

export interface HarnessInstallation {
  readonly executable: string
}

export interface HarnessConnector {
  readonly id: HarnessId
  readonly name: string
  readonly recommended?: boolean
  readonly note?: string
  readonly skillInstallationTarget: SkillInstallationTarget
  readonly configurationFiles: ReadonlyArray<string>
  readonly detect: (searchPath: string) => Effect.Effect<Option.Option<HarnessInstallation>>
  readonly connect: (
    spec: HarnessConnectionSpec,
  ) => Effect.Effect<void, unknown, FileSystem.FileSystem | Path.Path>
  readonly disconnect: (
    spec: HarnessConnectionSpec,
  ) => Effect.Effect<void, unknown, FileSystem.FileSystem | Path.Path>
  readonly launch: (
    modelId: ProviderModelId,
    installation: HarnessInstallation,
  ) => HarnessLaunchPlan
}
