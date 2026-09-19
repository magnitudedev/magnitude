import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Schedule, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure, Harness, InfrastructureFailure } from "../domain"
import { command, ProcessExecutor } from "../process"

export const CliTestConfig = Schema.Struct({ executable: Schema.String, version: Schema.String, model: Schema.String,
  evidence: Schema.String, environment: Schema.Record({ key: Schema.String, value: Schema.String }) })
export interface CliTests {
  readonly version: Effect.Effect<void, AssertionFailure | InfrastructureFailure>
  readonly inspect: Effect.Effect<void, AssertionFailure | InfrastructureFailure>
  readonly modelLifecycle: Effect.Effect<void, AssertionFailure | InfrastructureFailure>
  readonly connections: (harness: typeof Harness.Type, inspect: (connected: boolean) => Effect.Effect<void, AssertionFailure | InfrastructureFailure>) => Effect.Effect<void, AssertionFailure | InfrastructureFailure>
  readonly invalid: Effect.Effect<void, AssertionFailure | InfrastructureFailure>
  readonly nativeRuntime: Effect.Effect<void, AssertionFailure | InfrastructureFailure>
}
export const CliTests = Context.GenericTag<CliTests>("@magnitudedev/testing-lab/CliTests")
const fail = (message: string) => new AssertionFailure({ message })

export const bundledCliTests = (config: typeof CliTestConfig.Type) => Layer.effect(CliTests, Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executor = yield* ProcessExecutor
  yield* fs.makeDirectory(config.evidence, { recursive: true }).pipe(Effect.mapError(e => new InfrastructureFailure({ operation: "cli-evidence", message: e.message })))
  let index = 0
  const invoke = (args: readonly string[], env = config.environment) => Effect.gen(function* () {
    const result = yield* command(config.executable, args, { inheritEnv: false, env, timeoutMs: 180_000 }).pipe(Effect.provideService(ProcessExecutor, executor))
    const prefix = join(config.evidence, `${++index}-${args[0]?.replace(/[^a-z0-9-]/g, "") || "command"}`)
    yield* fs.writeFileString(`${prefix}.json`, yield* Schema.encode(Schema.parseJson(Schema.Struct({ args: Schema.Array(Schema.String), exitCode: Schema.Int,
      stdout: Schema.String, stderr: Schema.String })) )({ args, ...result })).pipe(Effect.mapError(e => new InfrastructureFailure({ operation: "cli-evidence", message: e.message })))
    return result
  }).pipe(Effect.mapError(error => new InfrastructureFailure({ operation: "cli-command", message: error.message })))
  const successful = (args: readonly string[]) => invoke(args).pipe(Effect.flatMap(result => result.exitCode === 0 ? Effect.succeed(result.stdout)
    : Effect.fail(fail(`Bundled CLI ${args.join(" ")} exited ${result.exitCode}: ${(result.stderr || result.stdout).slice(-1000)}`))))
  const assert = (condition: boolean, detail: string) => condition ? Effect.void : Effect.fail(fail(detail))
  return {
    version: successful(["--version"]).pipe(Effect.flatMap(output => assert(output.trim() === config.version, "Bundled CLI version differs from installed candidate"))),
    inspect: Effect.gen(function* () {
      const help = yield* successful(["--help"])
      yield* assert(["catalog", "models", "connections", "service", "hardware"].every(name => help.includes(name)), "Bundled CLI help omitted a supported command")
      const hardware = yield* successful(["hardware"])
      yield* assert(hardware.includes("Memory") && hardware.includes("CPU"), "Hardware command did not describe processor and memory")
      const service = yield* successful(["service", "status"])
      yield* assert(/Runtime\s+Ready/.test(service) && service.includes(config.version), "CLI did not observe the installed service ready at the accepted version")
    }),
    modelLifecycle: Effect.gen(function* () {
      for (const args of [["catalog", "list"], ["catalog", "show", config.model], ["models", "status", config.model]]) {
        yield* assert((yield* successful(args)).includes(config.model), `${args.join(" ")} did not identify the test model`)
      }
      const pull = yield* successful(["catalog", "pull", config.model])
      yield* assert(pull.includes("already installed and up to date"), "Cached model pull unexpectedly started another acquisition")
      yield* successful(["models", "stop"])
      yield* successful(["models", "load", config.model])
      yield* Effect.gen(function* () {
        const status = yield* successful(["models", "status", config.model])
        yield* assert(!/Runtime\s+Failed/.test(status), "CLI model load failed")
        return /Runtime\s+Ready/.test(status)
      }).pipe(Effect.repeat({ until: ready => ready, schedule: Schedule.spaced("1 second") }),
        Effect.timeoutFail({ duration: "3 minutes", onTimeout: () => fail("CLI model did not become ready") }))
      // Residency is not generation success: the caller must follow this with EndpointTests.generate.
    }),
    connections: (harness, inspect) => Effect.gen(function* () {
      yield* successful(["connections", "add", harness, "--set-model", config.model])
      yield* inspect(true)
      yield* successful(["connections", "sync", harness])
      yield* inspect(true)
      const connected = yield* successful(["connections", "list"])
      yield* assert(connected.split("\n").some(line => line.includes(harness) && /\bConnected\b/.test(line)), `CLI did not observe ${harness} connected`)
      yield* successful(["connections", "remove", harness])
      yield* inspect(false)
      const removed = yield* successful(["connections", "list"])
      yield* assert(removed.split("\n").some(line => line.includes(harness) && /\bDisconnected\b/.test(line)), `CLI did not observe ${harness} disconnected`)
      yield* successful(["connections", "add", harness, "--set-model", config.model])
      yield* inspect(true)
    }),
    invalid: Effect.gen(function* () {
      for (const args of [["lab-nonexistent-command"], ["catalog", "show", "lab-invalid-model"]]) {
        const result = yield* invoke(args)
        yield* assert(result.exitCode !== 0 && Boolean((result.stderr + result.stdout).trim()), "Invalid CLI input returned success or no diagnostic")
      }
      yield* assert(/Runtime\s+Ready/.test(yield* successful(["service", "status"])), "Invalid CLI input disrupted the owning service")
    }),
    nativeRuntime: Effect.gen(function* () {
      const path = process.platform === "win32" ? join(config.environment.SystemRoot ?? "C:\\Windows", "System32") : "/usr/bin:/bin:/usr/sbin:/sbin"
      const result = yield* invoke(["native-runtime-check"], { ...config.environment, PATH: path })
      yield* assert(result.exitCode === 0 && result.stdout.trim() === "Bun and SQLite native runtime ready", "Bundled CLI could not run its native database/runtime without developer tools on PATH")
    }),
  } satisfies CliTests
}))
