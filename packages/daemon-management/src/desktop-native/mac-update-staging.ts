import { UpdateRelease, verifyUpdateRelease } from "@magnitudedev/release/hosted-update"
import { Context, Effect, Option, Schema } from "effect"
import type { KeyObject } from "node:crypto"
import { GuardedCommand } from "./guarded-command"
import { MacUpdateFilesystem, type MacUpdateDirectory } from "./mac-update-filesystem"

export class MacUpdateStagingFailed extends Schema.TaggedError<MacUpdateStagingFailed>()("MacUpdateStagingFailed", {}) {
  override get message() { return "The application update could not be extracted and verified." }
}
export interface MacUpdateArchiveStager {
  readonly stage: (archive: string, staging: MacUpdateDirectory, release: UpdateRelease) => Effect.Effect<void, MacUpdateStagingFailed>
}
export const MacUpdateArchiveStager = Context.GenericTag<MacUpdateArchiveStager>("@magnitudedev/daemon-management/MacUpdateArchiveStager")

/** The installer retains private staging until the guarded extractor has completely retired. */
export const makeMacUpdateArchiveStager = (options: {
  readonly helper: string
  readonly architecture: "arm64" | "x64"
  readonly trustedPublishers: ReadonlyMap<string, KeyObject>
}) => Effect.gen(function* () {
  const fs = yield* MacUpdateFilesystem
  const runner = yield* GuardedCommand
  return MacUpdateArchiveStager.of({
    stage: (archive, staging, release) => Effect.gen(function* () {
      const authenticated = yield* verifyUpdateRelease(release,
        { os: "darwin", arch: options.architecture, package: "mac-zip" }, options.trustedPublishers)
      yield* fs.sync(staging)
      const result = yield* runner.run(options.helper,
        [archive, staging.path, authenticated.sha256, String(authenticated.bytes)], {}).pipe(Effect.timeout("5 minutes"))
      if (result.code !== 0) return yield* new MacUpdateStagingFailed()
      yield* fs.sync(staging)
      if (Option.isNone(yield* fs.inspect(staging, "Magnitude.app"))) return yield* new MacUpdateStagingFailed()
    }).pipe(Effect.mapError(() => new MacUpdateStagingFailed())),
  })
})
