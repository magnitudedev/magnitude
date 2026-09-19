import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { AssertionFailure } from "../domain"

export const FixtureFiles = Schema.Record({ key: Schema.String, value: Schema.String })
/** The model must change exactly one file. Control files are owned by the driver and immutable. */
export const fileFixture = (root: string, controls: typeof FixtureFiles.Type = {}) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectory({ directory: root, prefix: "harness-fixture-" }).pipe(Effect.flatMap(path => fs.realPath(path)))
  const files = { ...controls, "message.txt": "before\n", "untouched.txt": "This file must remain unchanged.\n" }
  for (const [name, content] of Object.entries(files)) {
    if (!/^[a-zA-Z0-9._-]+$/.test(name) || name === "." || name === "..") return yield* new AssertionFailure({ message: "Invalid fixture control file" })
    yield* fs.writeFileString(join(directory, name), content)
  }
  const verify = Effect.gen(function* () {
    const names = yield* fs.readDirectory(directory)
    if (names.slice().sort().join("\n") !== Object.keys(files).sort().join("\n")) return yield* new AssertionFailure({ message: "Harness added or removed an unexpected fixture file" })
    for (const [name, original] of Object.entries(files)) {
      const path = join(directory, name)
      if ((yield* fs.realPath(path)) !== path) return yield* new AssertionFailure({ message: `Harness replaced ${name} with a symlink` })
      const actual = yield* fs.readFileString(path)
      if (actual !== (name === "message.txt" ? "after\n" : original)) return yield* new AssertionFailure({ message: `Harness produced an unexpected change to ${name}` })
    }
  })
  return { directory, verify: verify.pipe(Effect.mapError(error => new AssertionFailure({ message: error.message }))) }
})
