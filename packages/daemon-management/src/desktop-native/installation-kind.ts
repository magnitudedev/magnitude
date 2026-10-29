import { FileSystem } from "@effect/platform"
import { LinuxInstallationKind } from "@magnitudedev/release/linux-package"
import { Effect, Option, Schema } from "effect"
import { join } from "node:path"

/**
 * The installed payload's delivery, from the same marker. Callers take container behaviour only when
 * this is positively `docker`; an unreadable marker reads as none and keeps package behaviour.
 */
export const readLinuxInstallationKind = (resources: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  return yield* fs.readFileString(join(resources, "update-package.json")).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ format: LinuxInstallationKind })))),
    Effect.map(marker => Option.some(marker.format)),
    Effect.orElseSucceed(() => Option.none<LinuxInstallationKind>()),
  )
})

export const isDockerInstallation = (resources: string) =>
  readLinuxInstallationKind(resources).pipe(Effect.map(kind => Option.contains(kind, "docker")))
