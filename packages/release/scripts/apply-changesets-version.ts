import * as Command from "@effect/platform/Command"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Effect } from "effect"
import { resolve } from "node:path"
import { prepareRelease } from "./prepare-release"

const root = resolve(import.meta.dir, "../../..")
const program = Effect.gen(function* () {
  const detection = yield* prepareRelease("detect")
  if (detection.pending) return
  // Changesets owns package versions in every channel; pre mode yields prerelease plugin versions too.
  const code = yield* Command.make("bunx", "changeset", "version").pipe(Command.workingDirectory(root), Command.stdin("inherit"), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
  if (code !== 0) return yield* Effect.dieMessage(`Changesets version failed (${code})`)
  // Allocation owns every derived identity, the daemon revision included.
  yield* prepareRelease("allocate")
  const lockCode = yield* Command.make("bun", "install", "--lockfile-only", "--ignore-scripts").pipe(Command.workingDirectory(root), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode)
  if (lockCode !== 0) return yield* Effect.dieMessage("Could not synchronize the release lockfile")
})
BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
