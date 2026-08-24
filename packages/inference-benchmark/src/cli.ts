#!/usr/bin/env bun
import { Args, Command } from "@effect/cli"
import * as PlatformCommand from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import * as FetchHttpClient from "@effect/platform/FetchHttpClient"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Data, Effect, Layer, Schema } from "effect"
import { join, resolve } from "node:path"
import { corpusStatus, prepareCorpus } from "./corpus"
import { discoverExperiments, loadExperiment, resolveExperimentPaths } from "./experiment"
import { compileTrialPlan, explainPlan } from "./plan"
import { createMlxArtifactLock, loadPreparedExperiment, prepareExperiment } from "./preparation"
import { activeRunLockPath, listRuns, runDirectory, runExperiment } from "./run"
import { TargetLauncherLive } from "./target"
import { EndpointClientLive } from "./transport"

class CliError extends Data.TaggedError("CliError")<{ readonly message: string }> {}
const operation = <A, E, R>(effect: Effect.Effect<A, E, R>): Effect.Effect<A, CliError, R> =>
  effect.pipe(Effect.mapError((error) => new CliError({ message: error instanceof Error ? error.message : String(error) })))

const experimentFile = Args.file({ name: "experiment", exists: "yes" })
const runId = Args.text({ name: "run-id" })
const RunProcessManifest = Schema.Struct({ runId: Schema.String, pid: Schema.Int })

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

const experimentsList = Command.make("list", {}, () => operation(Effect.gen(function* () {
  const entries = yield* discoverExperiments()
  for (const entry of entries) {
    const experiment = yield* loadExperiment(entry.path)
    yield* Console.log(`${experiment.id}\t${experiment.title}\t${entry.path}`)
  }
}))).pipe(Command.withDescription("List TypeScript experiment definitions"))

const experimentsShow = Command.make("show", { experiment: experimentFile }, ({ experiment }) => operation(Effect.gen(function* () {
  const loaded = yield* loadExperiment(experiment)
  const encoded = yield* Schema.encode(Schema.parseJson(Schema.Unknown, { space: 2 }))(resolveExperimentPaths(loaded, experiment))
  yield* Console.log(encoded)
}))).pipe(Command.withDescription("Validate and display a resolved experiment"))

const experiments = Command.make("experiments", {}).pipe(
  Command.withSubcommands([experimentsList, experimentsShow]),
  Command.withDescription("Inspect benchmark experiments"),
)

const modelsLockMlx = Command.make("lock-mlx", {
  repository: Args.text({ name: "repository" }),
  revision: Args.text({ name: "revision" }),
  bits: Args.integer({ name: "bits" }),
  groupSize: Args.integer({ name: "group-size" }),
  output: Args.text({ name: "output" }),
}, ({ repository, revision, bits, groupSize, output }) => operation(Effect.gen(function* () {
  const locked = yield* createMlxArtifactLock({ repository, revision, bits, groupSize, output })
  yield* Console.log(`locked: ${locked.output}\nfiles: ${locked.files}\ncache: ${locked.root}`)
}))).pipe(Command.withDescription("Download and write a complete immutable lock for an MLX artifact"))

const modelsLockMlxUnquantized = Command.make("lock-mlx-unquantized", {
  repository: Args.text({ name: "repository" }),
  revision: Args.text({ name: "revision" }),
  dtype: Args.text({ name: "dtype" }),
  output: Args.text({ name: "output" }),
}, ({ repository, revision, dtype, output }) => operation(Effect.gen(function* () {
  if (dtype !== "bfloat16" && dtype !== "float16") return yield* new CliError({ message: "dtype must be bfloat16 or float16" })
  const locked = yield* createMlxArtifactLock({ repository, revision, dtype, output })
  yield* Console.log(`locked: ${locked.output}\nfiles: ${locked.files}\ncache: ${locked.root}`)
}))).pipe(Command.withDescription("Download and lock an unquantized MLX artifact"))

const models = Command.make("models", {}).pipe(
  Command.withSubcommands([modelsLockMlx, modelsLockMlxUnquantized]),
  Command.withDescription("Create immutable model artifact locks"),
)

const prepare = Command.make("prepare", { experiment: experimentFile }, ({ experiment }) => operation(Effect.gen(function* () {
  const prepared = yield* prepareExperiment(experiment)
  yield* Console.log([
    `prepared: ${prepared.experiment.id}`,
    `digest: ${prepared.digest}`,
    ...prepared.artifacts.map((artifact) => `${artifact.variantId} (${artifact.role}): ${artifact.path}`),
  ].join("\n"))
}))).pipe(Command.withDescription("Download, verify, and freeze everything required by an experiment"))

const plan = Command.make("plan", { experiment: experimentFile }, ({ experiment }) => operation(Effect.gen(function* () {
  const loaded = resolveExperimentPaths(yield* loadExperiment(experiment), experiment)
  const prepared = yield* loadPreparedExperiment(loaded.id)
  const corpus = yield* prepareCorpus({ root: prepared.corpusRoot, offline: true })
  const compiled = yield* compileTrialPlan(corpus, prepared.planModel, {
    ...(loaded.suite.kind === "agent-core"
      ? { profile: loaded.suite.profile }
      : { contextSweep: loaded.suite }),
    maxContextTokens: loaded.requestPolicy.contextTokensPerSequence,
    parallelSequences: loaded.requestPolicy.parallelSequences,
    maxOutputTokens: loaded.requestPolicy.maxOutputTokens,
    temperature: loaded.requestPolicy.temperature,
    topP: loaded.requestPolicy.topP,
    seed: loaded.requestPolicy.seed,
    enableThinking: loaded.requestPolicy.enableThinking,
  })
  yield* Console.log(`${explainPlan(compiled)}\nblocks: ${loaded.execution.blocks}\norder: ${loaded.execution.variantOrder}`)
}))).pipe(Command.withDescription("Compile and display the immutable workload plan"))

const run = Command.make("run", { experiment: experimentFile }, ({ experiment }) => operation(Effect.gen(function* () {
  const result = yield* runExperiment(experiment)
  yield* Console.log(`completed: ${result.runId}\nresult: ${join(runDirectory(result.runId), "result.json")}`)
}))).pipe(Command.withDescription("Execute a prepared experiment"))

const runsList = Command.make("list", {}, () => operation(Effect.gen(function* () {
  const entries = yield* listRuns()
  yield* Console.log(entries.map((entry) => `${entry.runId}\t${entry.state}\t${entry.experimentId}\t${entry.startedAt}`).join("\n"))
})))

const runsShow = Command.make("show", { runId }, ({ runId }) => operation(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = runDirectory(runId)
  const result = join(directory, "result.json")
  const path = (yield* fs.exists(result)) ? result : join(directory, "manifest.json")
  yield* Console.log(yield* fs.readFileString(path))
})))

const runsWatch = Command.make("watch", { runId }, ({ runId }) => operation(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const eventsPath = join(runDirectory(runId), "events.jsonl")
  let delivered = 0
  while (true) {
    const text = yield* fs.readFileString(eventsPath)
    const lines = text.trim().length > 0 ? text.trimEnd().split("\n") : []
    for (const line of lines.slice(delivered)) yield* Console.log(line)
    delivered = lines.length
    if (lines.some((line) => /"type":"run-(completed|failed|cancelled)"/.test(line))) return
    yield* Effect.sleep("500 millis")
  }
})))

const runsCancel = Command.make("cancel", { runId }, ({ runId }) => operation(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const manifest = yield* Schema.decodeUnknown(Schema.parseJson(RunProcessManifest))(
    yield* fs.readFileString(join(runDirectory(runId), "manifest.json")),
  )
  const lock = yield* Schema.decodeUnknown(Schema.parseJson(RunProcessManifest))(
    yield* fs.readFileString(activeRunLockPath()),
  )
  if (manifest.runId !== runId || lock.runId !== runId || lock.pid !== manifest.pid) {
    return yield* new CliError({ message: "run manifest does not contain a valid matching PID" })
  }
  yield* Effect.try({
    try: () => process.kill(manifest.pid, "SIGINT"),
    catch: (error) => new CliError({ message: error instanceof Error ? error.message : String(error) }),
  })
  yield* Console.log(`cancellation requested for ${runId}`)
})))

const runs = Command.make("runs", {}).pipe(
  Command.withSubcommands([runsList, runsShow, runsWatch, runsCancel]),
  Command.withDescription("Inspect and control benchmark runs"),
)

const dashboard = Command.make("dashboard", {}, () => operation(Effect.gen(function* () {
  yield* Console.log("Inference benchmark dashboard: http://127.0.0.1:5187\nDashboard API: http://127.0.0.1:4897\nPress Ctrl+C to stop.")
  const child = PlatformCommand.make(
    "bunx", "concurrently", "--kill-others-on-fail", "bunx vite", "bun run src/server.ts",
  ).pipe(
    PlatformCommand.workingDirectory(resolve("packages/inference-benchmark-dashboard")),
    PlatformCommand.stdin("inherit"),
    PlatformCommand.stdout("inherit"),
    PlatformCommand.stderr("inherit"),
  )
  let interrupted = false
  const onInterrupt = () => { interrupted = true }
  const processEvents = process as unknown as NodeJS.EventEmitter
  const code = yield* Effect.acquireUseRelease(
    Effect.sync(() => processEvents.on("SIGINT", onInterrupt)),
    () => PlatformCommand.exitCode(child),
    () => Effect.sync(() => processEvents.removeListener("SIGINT", onInterrupt)),
  )
  if (code !== 0 && !interrupted) return yield* new CliError({ message: `dashboard exited with ${code}` })
}))).pipe(Command.withDescription("Start the local benchmark dashboard"))

const benchmark = Command.make("benchmark", {}).pipe(
  Command.withDescription("Magnitude inference benchmark"),
  Command.withSubcommands([corpus, models, experiments, prepare, plan, run, runs, dashboard]),
)

const cli = Command.run(benchmark, { name: "Magnitude inference benchmark", version: "0.2.0" })
const RuntimeLive = TargetLauncherLive.pipe(
  Layer.provideMerge(EndpointClientLive),
  Layer.provideMerge(FetchHttpClient.layer),
  Layer.provideMerge(BunContext.layer),
)
BunRuntime.runMain(cli(process.argv).pipe(Effect.provide(RuntimeLive)))
