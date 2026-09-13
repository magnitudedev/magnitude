import { FileSystem } from "@effect/platform"
import { UpdateClientMetadata } from "@magnitudedev/release/hosted-update"
import { Effect, Schema } from "effect"
import { join } from "node:path"

export const readLinuxUpdateMetadata = (resources: string, version: string, osVersion: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const format = yield* fs.readFileString(join(resources, "update-package.json")).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ format: Schema.Literal("deb", "rpm") })))),
  )
  const distro: Record<string, string> = {}
  for (const line of (yield* fs.readFileString("/etc/os-release")).split("\n")) {
    const field = /^(ID|VERSION_ID)=(?:"([^"\\]*)"|'([^'\\]*)'|([^\s'"\\]+))$/.exec(line)
    if (field) distro[field[1]!] = field[2] ?? field[3] ?? field[4]!
  }
  return yield* Schema.decodeUnknown(UpdateClientMetadata)({ version, os: "linux", os_version: osVersion,
    arch: process.arch, package: format.format, ...(distro.ID ? { distro: distro.ID } : {}), ...(distro.VERSION_ID ? { distro_version: distro.VERSION_ID } : {}) })
})
