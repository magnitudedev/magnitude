import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { join } from "node:path"
import { fileFixture } from "../src/harnesses/file-fixture"

test("file task accepts only the intended diff and rejects added files or changed controls", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-file-fixture-" })
  const task = yield* fileFixture(root, { "control.json": "{}\n" })
  expect((yield* task.verify.pipe(Effect.either))._tag).toBe("Left")
  yield* fs.writeFileString(join(task.directory, "message.txt"), "after\n")
  yield* task.verify
  yield* fs.writeFileString(join(task.directory, "extra.txt"), "unwanted\n")
  expect((yield* task.verify.pipe(Effect.either))._tag).toBe("Left")
  yield* fs.remove(join(task.directory, "extra.txt"))
  yield* fs.writeFileString(join(task.directory, "control.json"), "changed\n")
  expect((yield* task.verify.pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide(BunContext.layer))))
