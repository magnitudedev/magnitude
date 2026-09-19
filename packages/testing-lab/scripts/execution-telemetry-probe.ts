import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Config, Effect, Option, Schema } from "effect"
import { join, resolve } from "node:path"
import { executionTelemetry, NativeExecution } from "../src/execution-telemetry"
import { AssertionFailure } from "../src/domain"
import { command, CommandOutput, ProcessExecutorLive } from "../src/process"
import { assertRuntime } from "../src/runtime"

/** Real native exporter transport, with an explicitly synthetic completion; not backend qualification. */
BunRuntime.runMain(Effect.scoped(Effect.gen(function* () {
  yield* assertRuntime
  const fs = yield* FileSystem.FileSystem
  const root = resolve(yield* Config.string("LAB_TELEMETRY_PROBE_ROOT"))
  if (yield* fs.exists(root)) return yield* new AssertionFailure({ message: "Telemetry probe requires a fresh output directory" })
  yield* fs.makeDirectory(root, { recursive: true, mode: 0o700 })
  const collector = yield* executionTelemetry()
  const environment = Object.fromEntries(["PATH", "HOME", "TMPDIR", "SystemRoot", "TEMP"].flatMap(key => process.env[key] ? [[key, process.env[key]!]] : []))
  const result = yield* command("cargo", ["test", "-p", "icn-server", "--no-default-features", "telemetry::tests::exports_execution_fixture", "--", "--ignored", "--exact", "--nocapture"], {
    cwd: Option.some(resolve(import.meta.dir, "../../../inference")), inheritEnv: false,
    env: { ...environment, MAGNITUDE_OTEL_ENDPOINT: collector.endpoint }, timeoutMs: 30 * 60_000, maxOutputBytes: 8 * 1024 * 1024,
  })
  yield* fs.writeFileString(join(root, "native-exporter.json"), yield* Schema.encode(Schema.parseJson(CommandOutput))(result))
  if (result.exitCode !== 0) return yield* new AssertionFailure({ message: "Native exporter fixture failed; inspect native-exporter.json" })
  const observations = yield* collector.observations
  yield* fs.writeFileString(join(root, "observations.json"), yield* Schema.encode(Schema.parseJson(Schema.Array(NativeExecution)))(observations))
  if (observations.length !== 1 || observations[0]!.traceId !== "a".repeat(32) || observations[0]!.model !== "fixture-model") {
    return yield* new AssertionFailure({ message: "Native exporter did not deliver the exact synthetic request observation" })
  }
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))
