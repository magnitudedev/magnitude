#!/usr/bin/env bun
import { Args, Command, Options } from "@effect/cli"
import * as FileSystem from "@effect/platform/FileSystem"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Data, Effect, Layer } from "effect"
import { dirname, resolve } from "node:path"
import { compare, evaluate } from "./benchmark"
import { loadConfiguration } from "./configuration"
import { corpusStatus, prepareCorpus } from "./corpus"
import type { ModelIdentity, ProfileName } from "./domain"
import { resolveModelIdentity } from "./model"
import { listModelProfiles, resolveModelArtifact } from "./model-source"
import { compileTrialPlan, explainPlan } from "./plan"
import { renderBenchmarkMarkdown, renderComparisonMarkdown, writeReport } from "./report"
import {
  existingTarget,
  managedIcnTarget,
  managedLlamaCppTarget,
  resolveIcnExecutable,
  resolveLlamaCppExecutable,
  TargetLauncherLive,
} from "./target"
import { EndpointClientLive } from "./transport"

const profile = Options.choice("profile", ["smoke", "standard", "full"] as const).pipe(
  Options.withDefault("standard" as const),
)
const context = Options.integer("context").pipe(Options.withDefault(0))
const output = Options.text("output").pipe(Options.withDefault("benchmark-results/result.json"))
const modelPath = Options.text("model-path").pipe(
  Options.withDescription("Optional offline GGUF override"),
  Options.withDefault(""),
)
const apiKey = Options.text("api-key").pipe(Options.withDefault(""))
const modelReference = Args.text({ name: "model" })

class CliError extends Data.TaggedError("CliError")<{
  readonly message: string
}> {}

const operation = <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, CliError, R> =>
  effect.pipe(Effect.mapError((error) => new CliError({
    message: error instanceof Error ? error.message : String(error),
  })))

const requireSuite = (suite: string) => suite === "agent-core"
  ? Effect.void
  : Effect.fail(new CliError({ message: `Unknown suite ${suite}; expected agent-core` }))

const ensureOutputDirectory = (outputPath: string) =>
  Effect.flatMap(FileSystem.FileSystem, (fs) => fs.makeDirectory(dirname(outputPath), { recursive: true }))

const managedModel = (
  reference: string,
  localPath: string,
  contextLimit: number,
): Effect.Effect<ModelIdentity, CliError, FileSystem.FileSystem> =>
  operation(Effect.gen(function* () {
    const artifact = yield* resolveModelArtifact({
      reference,
      localPath: localPath || undefined,
      onDownload: (message) => console.error(message),
    })
    const identity = yield* resolveModelIdentity({
      id: artifact.id,
      artifactPath: artifact.path,
      maxContextTokens: contextLimit || undefined,
    })
    const expected = artifact.source.kind === "huggingface"
      ? artifact.source.expectedSha256
      : undefined
    if (expected && identity.artifactSha256 !== expected) {
      return yield* new CliError({
        message: `Model digest mismatch for ${reference}: expected ${expected}, received ${identity.artifactSha256}`,
      })
    }
    return identity
  }))

const corpusFetch = Command.make("fetch", {}, () => operation(Effect.gen(function* () {
  const prepared = yield* prepareCorpus()
  yield* Console.log(`BFCL V4 ready: ${prepared.fixtures.length} fixtures\ncache: ${prepared.root}\ndigest: ${prepared.digest}`)
}))).pipe(Command.withDescription("Download and verify the pinned BFCL V4 data"))

const corpusStatusCommand = Command.make("status", {}, () => operation(Effect.gen(function* () {
  const status = yield* corpusStatus()
  yield* Console.log(status.map((file) => `${file.valid ? "ok" : "missing"}  ${file.path}`).join("\n"))
}))).pipe(Command.withDescription("Show pinned BFCL cache status"))

const corpus = Command.make("corpus", {}).pipe(
  Command.withDescription("Manage the pinned BFCL corpus"),
  Command.withSubcommands([corpusFetch, corpusStatusCommand]),
)

const modelsList = Command.make("list", {}, () => Console.log(
  listModelProfiles().map((entry) => [
    entry.id,
    entry.displayName,
    `${(entry.sizeBytes / 1_000_000_000).toFixed(1)} GB`,
    `${entry.repository}/resolve/${entry.revision}/${entry.file}`,
  ].join("\t")).join("\n"),
)).pipe(Command.withDescription("List built-in immutable model profiles"))

const modelsFetch = Command.make(
  "fetch",
  { model: modelReference, modelPath },
  ({ model, modelPath }) => operation(Effect.gen(function* () {
    const identity = yield* managedModel(model, modelPath, 0)
    yield* Console.log(`model: ${identity.artifactPath}\nsha256: ${identity.artifactSha256}`)
  })),
).pipe(Command.withDescription("Resolve, download, cache, and verify a model"))

const models = Command.make("models", {}).pipe(
  Command.withDescription("Manage benchmark model artifacts"),
  Command.withSubcommands([modelsList, modelsFetch]),
)

const explain = Command.make(
  "explain",
  { suite: Args.text({ name: "suite" }), model: modelReference, modelPath, context, profile },
  ({ suite, model, modelPath, context, profile }) => operation(Effect.gen(function* () {
    yield* requireSuite(suite)
    const prepared = yield* prepareCorpus()
    const identity = yield* managedModel(model, modelPath, context)
    const plan = yield* compileTrialPlan(prepared, identity, { profile: profile as ProfileName })
    yield* Console.log(explainPlan(plan))
  })),
).pipe(Command.withDescription("Compile and explain an immutable benchmark plan"))

const endpoint = Options.text("endpoint")
const targetId = Options.text("target").pipe(Options.withDefault("endpoint"))
const maxSequences = Options.integer("sequences").pipe(Options.withDefault(4))

const run = Command.make(
  "run",
  { suite: Args.text({ name: "suite" }), model: modelReference, modelPath, endpoint, targetId, apiKey, context, profile, output, maxSequences },
  ({ suite, model, modelPath, endpoint, targetId, apiKey, context, profile, output, maxSequences }) => operation(Effect.gen(function* () {
    yield* requireSuite(suite)
    const prepared = yield* prepareCorpus()
    const identity = yield* managedModel(model, modelPath, context)
    const plan = yield* compileTrialPlan(prepared, identity, {
      profile: profile as ProfileName,
      parallelSequences: maxSequences,
    })
    const result = yield* evaluate(
      plan,
      existingTarget(targetId, endpoint, model, apiKey || undefined, maxSequences),
    )
    const destination = resolve(output)
    yield* ensureOutputDirectory(destination)
    yield* writeReport(destination, result)
    yield* Console.log(`${renderBenchmarkMarkdown(result)}\n\nJSON: ${destination}`)
  })),
).pipe(Command.withDescription("Evaluate one conforming chat-completions endpoint"))

const icnExecutable = Options.text("icn-executable").pipe(Options.withDefault(""))
const llamaExecutable = Options.text("llama-executable").pipe(Options.withDefault(""))
const port = Options.integer("port").pipe(Options.withDefault(8091))

const compareCommand = Command.make(
  "compare",
  {
    suite: Args.text({ name: "suite" }), model: modelReference, modelPath,
    icnExecutable, llamaExecutable, port, maxSequences, context, profile, output,
  },
  ({ suite, model, modelPath, icnExecutable, llamaExecutable, port, maxSequences, context, profile, output }) =>
    operation(Effect.gen(function* () {
      yield* requireSuite(suite)
      const identity = yield* managedModel(model, modelPath, context)
      const prepared = yield* prepareCorpus()
      const plan = yield* compileTrialPlan(prepared, identity, {
        profile: profile as ProfileName,
        parallelSequences: maxSequences,
      })
      const resolvedIcn = yield* resolveIcnExecutable(icnExecutable || undefined)
      const resolvedLlama = yield* resolveLlamaCppExecutable(llamaExecutable || undefined)
      const options = {
        model: identity,
        icnExecutable: resolvedIcn,
        llamaExecutable: resolvedLlama,
        port,
        maxSequences,
      }
      const result = yield* compare(plan, [managedIcnTarget(options), managedLlamaCppTarget(options)])
      const destination = resolve(output)
      yield* ensureOutputDirectory(destination)
      yield* writeReport(destination, result)
      yield* Console.log(`${renderComparisonMarkdown(result)}\n\nJSON: ${destination}`)
    })),
).pipe(Command.withDescription("Compare managed ICN and llama.cpp with equal declared capacity"))

const execute = Command.make(
  "execute",
  { config: Args.file({ name: "config", exists: "yes" }) },
  ({ config }) => operation(Effect.gen(function* () {
    const configuration = yield* loadConfiguration(config)
    const prepared = yield* prepareCorpus({ root: configuration.corpusCache })
    const model = yield* resolveModelIdentity({
      id: configuration.model.id,
      artifactPath: resolve(configuration.model.artifactPath),
      maxContextTokens: configuration.model.contextLimit,
    })
    if (configuration.model.artifactSha256 && model.artifactSha256 !== configuration.model.artifactSha256) {
      return yield* new CliError({
        message: `Model digest mismatch: expected ${configuration.model.artifactSha256}, received ${model.artifactSha256}`,
      })
    }
    const capacities = new Set(configuration.targets.map((target) => target.parallelSequences))
    if (capacities.size !== 1) {
      return yield* new CliError({ message: "all configured targets must declare the same parallelSequences" })
    }
    const plan = yield* compileTrialPlan(prepared, model, {
      profile: configuration.profile,
      parallelSequences: configuration.targets[0]!.parallelSequences,
    })
    const destination = resolve(configuration.output)
    yield* ensureOutputDirectory(destination)
    if (configuration.targets.length === 1) {
      const result = yield* evaluate(plan, configuration.targets[0]!)
      yield* writeReport(destination, result)
      yield* Console.log(renderBenchmarkMarkdown(result))
    } else {
      const result = yield* compare(plan, configuration.targets)
      yield* writeReport(destination, result)
      yield* Console.log(renderComparisonMarkdown(result))
    }
    yield* Console.log(`\nJSON: ${destination}`)
  })),
).pipe(Command.withDescription("Execute a JSON benchmark configuration"))

const benchmark = Command.make("benchmark", {}).pipe(
  Command.withDescription("Magnitude inference benchmark"),
  Command.withSubcommands([run, compareCommand, execute, explain, corpus, models]),
)

const cli = Command.run(benchmark, { name: "Magnitude inference benchmark", version: "0.1.0" })
const RuntimeLive = TargetLauncherLive.pipe(
  Layer.provideMerge(EndpointClientLive),
  Layer.provideMerge(FetchHttpClient.layer),
  Layer.provideMerge(BunContext.layer),
)
BunRuntime.runMain(cli(process.argv).pipe(Effect.provide(RuntimeLive)))
