import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { HarnessTools, harnessSuite } from "../src/harnesses/suite"
import { ProcessExecutor } from "../src/process"

for (const version of ["1.18.310", "1.18.30", "unexpected"]) test(`rejects unqualified OpenCode version ${version} before generation`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-harness-version-" })
  let calls = 0
  const result = yield* harnessSuite("opencode", "fixture", join(root, "suite"), join(root, "home"), {}).pipe(
    Effect.provideService(HarnessTools, { executable: () => Effect.succeed("/qualified/opencode") }),
    Effect.provideService(ProcessExecutor, { run: spec => Effect.sync(() => {
      calls++
      expect(spec.args).toEqual(["--version"])
      expect(spec.inheritEnv).toBe(false)
      expect(spec.env.HOME).toBe(join(root, "home"))
      return { exitCode: 0, stdout: version, stderr: "" }
    }) }), Effect.either)
  expect(result._tag === "Left" && result.left._tag).toBe("InfrastructureFailure")
  expect(calls).toBe(1)
})).pipe(Effect.provide(BunContext.layer))))
