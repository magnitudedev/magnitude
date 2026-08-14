import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Option, Schema } from "effect"
import type { ProfileName, TargetConfiguration } from "./domain"

const NonEmptyString = Schema.String.pipe(Schema.minLength(1))
const PositiveInt = Schema.Int.pipe(Schema.greaterThan(0))
const optionalString = Schema.optionalWith(NonEmptyString, { as: "Option", exact: true })
const optionalNumber = Schema.optionalWith(PositiveInt, { as: "Option", exact: true })
const optionalUnknownRecord = Schema.optionalWith(
  Schema.Record({ key: Schema.String, value: Schema.Unknown }),
  { as: "Option", exact: true },
)

const ExistingTargetConfiguration = Schema.Struct({
  kind: Schema.Literal("existing"),
  id: NonEmptyString,
  endpoint: NonEmptyString,
  servedModel: NonEmptyString,
  apiKey: optionalString,
  requestBody: optionalUnknownRecord,
  parallelSequences: PositiveInt,
})

const ModelLoadConfiguration = Schema.Struct({
  artifactSha256: NonEmptyString,
  contextLimit: PositiveInt,
  instanceId: NonEmptyString,
})

const ManagedTargetConfiguration = Schema.Struct({
  kind: Schema.Literal("managed"),
  engine: Schema.Literal("icn", "llama.cpp", "vllm", "sglang", "generic"),
  id: NonEmptyString,
  executable: NonEmptyString,
  args: Schema.Array(Schema.String),
  endpoint: NonEmptyString,
  servedModel: NonEmptyString,
  parallelSequences: PositiveInt,
  apiKey: optionalString,
  requestBody: optionalUnknownRecord,
  env: Schema.optionalWith(
    Schema.Record({ key: Schema.String, value: Schema.String }),
    { as: "Option", exact: true },
  ),
  readinessPath: optionalString,
  cwd: optionalString,
  modelLoad: Schema.optionalWith(ModelLoadConfiguration, { as: "Option", exact: true }),
})

const BenchmarkConfiguration = Schema.Struct({
  suite: Schema.Literal("agent-core"),
  profile: Schema.Literal("smoke", "standard", "full"),
  model: Schema.Struct({
    id: NonEmptyString,
    artifactPath: NonEmptyString,
    contextLimit: PositiveInt,
    artifactSha256: optionalString,
  }),
  targets: Schema.NonEmptyArray(Schema.Union(ExistingTargetConfiguration, ManagedTargetConfiguration)),
  output: NonEmptyString,
  corpusCache: optionalString,
})

type DecodedConfiguration = typeof BenchmarkConfiguration.Type

export interface ModelConfiguration {
  readonly id: string
  readonly artifactPath: string
  readonly contextLimit: number
  readonly artifactSha256?: string
}

export interface BenchmarkFileConfiguration {
  readonly suite: "agent-core"
  readonly profile: ProfileName
  readonly model: ModelConfiguration
  readonly targets: readonly TargetConfiguration[]
  readonly output: string
  readonly corpusCache?: string
}

export class ConfigurationError extends Data.TaggedError("ConfigurationError")<{
  readonly path: string
  readonly message: string
}> {}

const targetFromDecoded = (
  target: DecodedConfiguration["targets"][number],
): TargetConfiguration => target.kind === "existing"
  ? {
      kind: target.kind,
      id: target.id,
      endpoint: target.endpoint,
      servedModel: target.servedModel,
      apiKey: Option.getOrUndefined(target.apiKey),
      requestBody: Option.getOrUndefined(target.requestBody),
      parallelSequences: target.parallelSequences,
    }
  : {
      kind: target.kind,
      engine: target.engine,
      id: target.id,
      executable: target.executable,
      args: target.args,
      endpoint: target.endpoint,
      servedModel: target.servedModel,
      parallelSequences: target.parallelSequences,
      apiKey: Option.getOrUndefined(target.apiKey),
      requestBody: Option.getOrUndefined(target.requestBody),
      env: Option.getOrUndefined(target.env),
      readinessPath: Option.getOrUndefined(target.readinessPath),
      cwd: Option.getOrUndefined(target.cwd),
      modelLoad: Option.getOrUndefined(target.modelLoad),
    }

const toConfiguration = (decoded: DecodedConfiguration): BenchmarkFileConfiguration => ({
  suite: decoded.suite,
  profile: decoded.profile,
  model: {
    id: decoded.model.id,
    artifactPath: decoded.model.artifactPath,
    contextLimit: decoded.model.contextLimit,
    artifactSha256: Option.getOrUndefined(decoded.model.artifactSha256),
  },
  targets: decoded.targets.map(targetFromDecoded),
  output: decoded.output,
  corpusCache: Option.getOrUndefined(decoded.corpusCache),
})

export const loadConfiguration = (
  path: string,
): Effect.Effect<BenchmarkFileConfiguration, ConfigurationError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const text = yield* fs.readFileString(path)
    const decoded = yield* Schema.decodeUnknown(Schema.parseJson(BenchmarkConfiguration))(text)
    return toConfiguration(decoded)
  }).pipe(Effect.mapError((error) => new ConfigurationError({
    path,
    message: error instanceof Error ? error.message : String(error),
  })))
