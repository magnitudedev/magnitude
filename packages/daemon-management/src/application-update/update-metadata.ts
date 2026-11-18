import { FileSystem } from "@effect/platform"
import { UpdateClientMetadata, type UpdateOwner } from "@magnitudedev/release/hosted-update"
import { LinuxInstallationKind } from "@magnitudedev/release/linux-package"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import type { InstallationLayout } from "../desktop-native/application-bootstrap"

const PackageMarker = Schema.parseJson(Schema.Struct({ format: LinuxInstallationKind }))

/**
 * A desktop package names its format in `update-package.json`. A standalone installation is `cli-only`,
 * unless its marker says it is a Docker image.
 */
export const readLinuxUpdateMetadata = (resources: string, layout: InstallationLayout, version: string, osVersion: string, owner: UpdateOwner) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const marker = fs.readFileString(join(resources, "update-package.json")).pipe(Effect.flatMap(Schema.decodeUnknown(PackageMarker)))
  const format = layout === "Desktop" ? yield* marker
    : yield* marker.pipe(Effect.map(read => read.format === "docker" ? read : { format: "cli-only" as const }),
      Effect.catchIf(error => error._tag === "SystemError" && error.reason === "NotFound", () => Effect.succeed({ format: "cli-only" as const })))
  const distro: Record<string, string> = {}
  for (const line of (yield* fs.readFileString("/etc/os-release")).split("\n")) {
    const field = /^(ID|VERSION_ID)=(?:"([^"\\]*)"|'([^'\\]*)'|([^\s'"\\]+))$/.exec(line)
    if (field) distro[field[1]!] = field[2] ?? field[3] ?? field[4]!
  }
  return yield* Schema.decodeUnknown(UpdateClientMetadata)({ version, os: "linux", os_version: osVersion, owner,
    arch: process.arch, package: format.format, ...(distro.ID ? { distro: distro.ID } : {}), ...(distro.VERSION_ID ? { distro_version: distro.VERSION_ID } : {}) })
})
