import { FetchHttpClient, FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Console, Effect, Layer, Option, Schema, Stream } from "effect"
import { join, resolve } from "node:path"
import { homedir } from "node:os"
import { LabClient, labClientLayer } from "./client"
import { planRun, targets } from "./catalog"
import { Digest, InfrastructureFailure, InvalidInput, RunId, RunPlan, RunRequest, RunResult, Target, resultExitCode } from "./domain"
import { ProcessExecutorLive } from "./process"
import { manifestJson, snapshotSource } from "./snapshot"
import { assertRuntime } from "./runtime"
import { RunRecord } from "./run-store"

const help = `Magnitude testing lab

  bun lab targets
  bun lab plan --request run.json
  bun lab run --source . --target macos-26-arm64-metal-apple-silicon
  bun lab run --source . --profile pr --budget 150 --concurrency 4
  bun lab status --run run-<uuid>
  bun lab results --run run-<uuid>
  bun lab cancel --run run-<uuid>

Run uploads dirty tracked files and nonignored new files without a commit or push.
LAB_URL and LAB_TOKEN select the authenticated coordinator. Token identity determines
ownership and trust. --mode iterate|verify defaults to verify. --no-wait submits and
returns the run ID. --allow-spark is explicit consent to use the shared office Spark.
Results exit 0 only when every selected case passed and cleanup completed.
`
export const parseArguments = (args: readonly string[]) => Effect.gen(function* () {
  const command = args[0] ?? "help"
  const options = new Map<string, string>()
  const boolean = new Set(["no-wait", "allow-spark"])
  const allowed = new Set(["request", "source", "target", "profile", "budget", "concurrency", "deadline", "mode", "objects", "run", ...boolean])
  for (let i = 1; i < args.length; i++) {
    const flag = args[i]!
    if (!flag.startsWith("--") || !allowed.has(flag.slice(2))) return yield* new InvalidInput({ message: `Unknown argument: ${flag}` })
    const name = flag.slice(2)
    if (options.has(name)) return yield* new InvalidInput({ message: `Duplicate option: ${flag}` })
    const value = boolean.has(name) ? "true" : args[++i]
    if (!value || value.startsWith("--")) return yield* new InvalidInput({ message: `Missing value: ${flag}` })
    options.set(name, value)
  }
  return { command, options }
})
const print = <A, I>(schema: Schema.Schema<A, I>, value: A) => Schema.encode(Schema.parseJson(schema))(value).pipe(Effect.flatMap(Console.log))
const remote = <A, E, R>(effect: Effect.Effect<A, E, R | LabClient>) => Effect.gen(function* () {
  const url = yield* Config.string("LAB_URL")
  const token = yield* Config.redacted("LAB_TOKEN")
  return yield* effect.pipe(Effect.provide(labClientLayer(url, token).pipe(Layer.provide(FetchHttpClient.layer))))
})
export const cli = (args: readonly string[]) => Effect.gen(function* () {
  yield* assertRuntime
  const { command, options } = yield* parseArguments(args)
  const fs = yield* FileSystem.FileSystem
  const required = (name: string) => Effect.fromNullable(options.get(name)).pipe(Effect.mapError(() => new InvalidInput({ message: `Missing --${name}` })))
  if (command === "help" || command === "--help") return yield* Console.log(help)
  if (command === "targets") return yield* print(Schema.Array(Target), targets)
  if (command === "plan") {
    const request = yield* fs.readFileString(yield* required("request")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(RunRequest))))
    return yield* print(RunPlan, yield* planRun(request))
  }
  if (["status", "results", "cancel"].includes(command)) {
    const id = yield* Schema.decodeUnknown(RunId)(yield* required("run"))
    return yield* remote(Effect.gen(function* () {
      const client = yield* LabClient
      if (command === "status") return yield* print(RunRecord, yield* client.get(id))
      if (command === "cancel") return yield* print(RunRecord, yield* client.cancel(id))
      const result = yield* client.result(id)
      if (Option.isNone(result)) { yield* Console.error("Run is still in progress"); process.exitCode = 2; return }
      yield* print(RunResult, result.value)
      process.exitCode = resultExitCode(result.value)
    }))
  }
  if (command !== "run") return yield* new InvalidInput({ message: `Unknown command: ${command}` })
  const source = resolve(yield* required("source"))
  const objects = resolve(options.get("objects") ?? join(homedir(), ".cache", "magnitude-lab", "objects"))
  return yield* remote(Effect.gen(function* () {
    const client = yield* LabClient
    const identity = yield* client.identity()
    const snapshot = yield* snapshotSource(source, objects)
    const request = yield* Schema.decodeUnknown(RunRequest)({ schemaVersion: 1, idempotencyKey: crypto.randomUUID(), ...identity,
      input: { kind: "source", digest: snapshot.digest }, selection: { kind: "profile", profile: options.get("profile") ?? "quick", ...(options.has("target") ? { target: options.get("target") } : {}) },
      mode: options.get("mode") ?? "verify", allowSpark: options.has("allow-spark"),
      limits: { concurrency: Number(options.get("concurrency") ?? 1), deadlineMinutes: Number(options.get("deadline") ?? 60), budgetUsd: Number(options.get("budget") ?? 25), idleMinutes: 15 },
    })
    const plan = yield* client.plan(request)
    yield* Console.error(`Planned ${plan.targets.length} targets; reserved estimate $${plan.estimatedComputeUsd}; source ${snapshot.digest}`)
    if (plan.estimatedComputeUsd > request.limits.budgetUsd) return yield* new InvalidInput({ message: "Plan exceeds --budget; no input uploaded or run submitted" })
    const digests = [...new Set(snapshot.manifest.entries.flatMap(e => e.kind === "file" ? [e.sha256] : []))]
    const missing: Digest[] = []
    for (let start = 0; start < digests.length; start += 1000) missing.push(...yield* client.missing(digests.slice(start, start + 1000)))
    yield* Console.error(`Uploading ${missing.length} changed objects; ${digests.length - missing.length} already available`)
    yield* Effect.forEach(missing, digest => client.upload(digest, fs.stream(join(objects, digest)).pipe(Stream.mapError(() => new InfrastructureFailure({ operation: "source-upload", message: "Cannot read source object" })))), { concurrency: 4, discard: true })
    yield* client.upload(snapshot.digest, Stream.make(new TextEncoder().encode(manifestJson(snapshot.manifest))))
    yield* client.registerInput(request.input)
    const run = yield* client.submit(request)
    yield* print(RunRecord, run)
    if (options.has("no-wait")) return
    yield* Console.error(`Waiting for ${run.state.runId}; interrupting this client leaves the remote run active. Use lab cancel to stop it.`)
    for (;;) {
      const result = yield* client.result(run.state.runId)
      if (Option.isSome(result)) {
        yield* print(RunResult, result.value)
        process.exitCode = resultExitCode(result.value)
        return
      }
      yield* Effect.sleep("3 seconds")
    }
  }))
})
if (import.meta.main) BunRuntime.runMain(cli(process.argv.slice(2)).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
