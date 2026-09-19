import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer } from "effect"
import { expect, test } from "vitest"
import { AssertionFailure } from "../src/domain"
import { ProcessExecutor } from "../src/process"
import { bundledCliTests, CliTests } from "../src/suites/cli"

for (const failureAt of [0, 1, 2, 3, 4]) test(`CLI connection file verification ${failureAt ? `rejects incorrect stage ${failureAt}` : "checks every mutation"}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-cli-connections-" })
  let connected = false
  const checks: boolean[] = []
  const commands: string[][] = []
  const executor = Layer.succeed(ProcessExecutor, { run: spec => Effect.sync(() => {
    expect(spec.executable).toBe("/installed/magnitude")
    expect(spec.inheritEnv).toBe(false)
    commands.push([...spec.args])
    if (spec.args[1] === "add") connected = true
    if (spec.args[1] === "remove") connected = false
    return { exitCode: 0, stderr: "", stdout: `pi ${connected ? "Connected" : "Disconnected"}` }
  }) })
  const result = yield* CliTests.pipe(Effect.flatMap(cli => cli.connections("pi", value => {
    checks.push(value)
    return checks.length === failureAt ? Effect.fail(new AssertionFailure({ message: "Actual configuration is wrong" })) : Effect.void
  })), Effect.provide(bundledCliTests({ executable: "/installed/magnitude", version: "0.1.3", model: "test-model", evidence: root, environment: {} }).pipe(Layer.provide(executor))), Effect.either)
  expect(result._tag).toBe(failureAt ? "Left" : "Right")
  expect(checks).toEqual([true, true, false, true].slice(0, failureAt || 4))
  if (failureAt === 1) expect(commands).toHaveLength(1)
})).pipe(Effect.provide(BunContext.layer))))
