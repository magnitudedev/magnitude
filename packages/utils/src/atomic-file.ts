import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect } from "effect"
import { randomUUID } from "node:crypto"

/**
 * A name beside `file` no other writer holds, as the inference engine's stores
 * (`seismic_io::store`) name a write under way: `.<name>.<pid>.<unique>.tmp`.
 * A store whose root holds `file` removes one a dead writer left once it is
 * abandoned.
 */
const temporaryName = (path: Path.Path, file: string) =>
  path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${randomUUID()}.tmp`)

/** Replace a user-owned configuration file without exposing partial contents. */
export const writeFileAtomic = (file: string, contents: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  yield* fs.makeDirectory(path.dirname(file), { recursive: true, mode: 0o700 })
  const temporary = temporaryName(path, file)
  // The original, kept aside while the replacement lands, is a real file: a name outside the
  // temporary convention, so no sweep removes the only copy a crash between the renames leaves.
  const backup = `${file}.${process.pid}.${randomUUID()}.bak`
  yield* fs.writeFileString(temporary, contents, { mode: 0o600 })
  yield* fs.rename(temporary, file).pipe(
    Effect.catchAll((replacementError) => Effect.gen(function* () {
      // Windows cannot rename over an existing file. Keep the original until
      // the replacement has succeeded so a second failure cannot destroy it.
      if (!(yield* fs.exists(file))) return yield* replacementError
      yield* fs.rename(file, backup)
      yield* fs.rename(temporary, file).pipe(
        Effect.catchAll((error) => fs.rename(backup, file).pipe(
          Effect.zipRight(Effect.fail(error)),
        )),
      )
      yield* fs.remove(backup).pipe(Effect.ignore)
    })),
    Effect.ensuring(fs.remove(temporary).pipe(Effect.ignore)),
  )
})
