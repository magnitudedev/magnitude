import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect } from "effect"
import { randomUUID } from "node:crypto"

/** Replace a user-owned configuration file without exposing partial contents. */
export const writeFileAtomic = (file: string, contents: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const path = yield* Path.Path
  yield* fs.makeDirectory(path.dirname(file), { recursive: true, mode: 0o700 })
  const occurrence = `${process.pid}.${randomUUID()}`
  const temporary = `${file}.${occurrence}.tmp`
  const backup = `${file}.${occurrence}.bak`
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
