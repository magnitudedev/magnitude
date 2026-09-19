import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { expect, test } from "vitest"
import { bundledCliTests, CliTests } from "../src/suites/cli"
import { ProcessExecutor } from "../src/process"

for (const fault of ["none", "stop", "load"] as const) test(`model reload observes stop and rejects ${fault} failures`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const evidence = yield* fs.makeTempDirectoryScoped({ prefix: "lab-model-reload-" })
  const calls: string[] = []
  let stopped = false, loaded = false
  const executor = Layer.succeed(ProcessExecutor, { run: spec => Effect.sync(() => {
    const operation = spec.args[1]!
    calls.push(operation)
    if (operation === "stop") stopped = true
    if (operation === "load") { expect(stopped).toBe(true); loaded = true }
    return { exitCode: 0, stderr: "", stdout: operation !== "status" ? "Requested" : `Runtime ${!loaded ? fault === "stop" ? "Failed" : "Unloaded" : fault === "load" ? "Failed" : "Ready"}` }
  }) })
  const result = yield* CliTests.pipe(Effect.flatMap(cli => cli.reloadModel), Effect.provide(bundledCliTests({
    executable: "/installed/magnitude", version: "0.1.3", model: "model", evidence, environment: {},
  }).pipe(Layer.provide(executor))), Effect.either)
  expect(result._tag).toBe(fault === "none" ? "Right" : "Left")
  expect(calls).toEqual(fault === "stop" ? ["stop", "status"] : ["stop", "status", "load", "status"])
})).pipe(Effect.provide(BunContext.layer))))
