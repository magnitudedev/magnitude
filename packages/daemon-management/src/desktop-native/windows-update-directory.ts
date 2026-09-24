import { FileSystem } from "@effect/platform"
import { Effect, Schema } from "effect"
import { createRequire } from "node:module"
import { join } from "node:path"

export class WindowsUpdateDirectoryFailed extends Schema.TaggedError<WindowsUpdateDirectoryFailed>()("WindowsUpdateDirectoryFailed", {
  message: Schema.String,
}) {}

const failed = () => new WindowsUpdateDirectoryFailed({
  message: "The update directory could not be prepared safely. Existing files were preserved. Check the updates directory permissions and contents before trying again.",
})

/** Run before reading prepared state, under exclusive application or installation ownership. */
export const recoverWindowsUpdateDirectory = (addonPath: string, dataDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(dataDirectory, { recursive: true })
  return yield* Effect.try({
    try: () => {
      const native = createRequire(import.meta.url)(addonPath) as { readonly recoverUpdateDirectory: (path: string) => unknown }
      return native.recoverUpdateDirectory(join(dataDirectory, "updates"))
    },
    catch: failed,
  }).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.Boolean)))
}).pipe(Effect.mapError(failed))
