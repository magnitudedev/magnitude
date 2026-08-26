import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import type {
  HarnessId,
  HarnessLaunchPlan,
} from "@magnitudedev/client-common"
import { ProviderModelIdSchema, type ProviderModelId } from "@magnitudedev/sdk"
import { Schema, type Effect, type Option } from "effect"

export const HarnessModelSchema = Schema.Struct({
  id: ProviderModelIdSchema,
  name: Schema.NonEmptyString,
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
  readonly installSkill: (
    contents: string,
  ) => Effect.Effect<void, unknown, FileSystem.FileSystem | Path.Path>
}
