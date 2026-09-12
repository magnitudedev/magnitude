import { writeAppleConsumerReceipt } from "./apple/distribution"
import { runAppleBuild } from "./apple/compile-bun"
import { fileSha256 } from "./build/common"
import { readFile, stat } from "node:fs/promises"
import { resolve } from "node:path"
import { Effect, Schema } from "effect"
import { BunContext } from "@effect/platform-bun"
import { validateLinuxDesktopInstaller } from "./build/desktop-linux"
import { ACN_COORDINATION_REVISION } from "@magnitudedev/version"
import { ReleaseArtifactSchema } from "../src/contracts"
import { acnArchive, cliArchive, hostById, icnBaseArchive, type HostId } from "../src/targets"
import { smokeHostArchives } from "./build/host"

const hostId = process.argv[2] as HostId | undefined
if (hostId === undefined) {
  throw new Error("usage: validate-host.ts <host-id> <artifact-directory>")
}
const host = hostById(hostId)
const root = resolve(process.argv[3] ?? `release/${hostId}`)
const artifact = Schema.decodeUnknownSync(Schema.parseJson(ReleaseArtifactSchema))(
  await readFile(resolve(root, `icn-base-${hostId}.artifact.json`), "utf8")
)

const ids = ["cli", "acn", "icn-base"].map(kind => `${kind}-${hostId}`)
if (hostId.startsWith("darwin-")) ids.push(`desktop-${hostId}`, `desktop-update-${hostId}`)
if (hostId.startsWith("linux-")) ids.push(`desktop-${hostId}-deb`, `desktop-${hostId}-rpm`)
const artifacts = await Promise.all(ids.map(async (id) => {
  const metadata = Schema.decodeUnknownSync(Schema.parseJson(ReleaseArtifactSchema))(await readFile(resolve(root, `${id}.artifact.json`), "utf8"))
  if (metadata.id !== id || Number((await stat(resolve(root, metadata.filename))).size) !== metadata.bytes || await fileSha256(resolve(root, metadata.filename)) !== metadata.sha256) throw new Error(`Consumer received changed ${id} bytes`)
  return metadata
}))

if (hostId === "linux-arm64-gnu" || hostId === "linux-x64-gnu") {
  const { version } = Schema.decodeUnknownSync(Schema.parseJson(Schema.Struct({ version: Schema.NonEmptyString })))(
    await readFile(resolve(import.meta.dir, "../../launcher/package.json"), "utf8"),
  )
  for (const format of ["deb", "rpm"] as const) {
    const installer = artifacts.find(value => value.id === `desktop-${hostId}-${format}`)!
    await Effect.runPromise(validateLinuxDesktopInstaller({ file: resolve(root, installer.filename), format,
      arch: hostId === "linux-arm64-gnu" ? "arm64" : "x64", version, revision: ACN_COORDINATION_REVISION,
    }).pipe(Effect.provide(BunContext.layer)))
  }
}

await smokeHostArchives(
  host,
  resolve(root, cliArchive(hostId)),
  resolve(root, acnArchive(hostId)),
  resolve(root, icnBaseArchive(hostId)),
  artifact
)

if (hostId.startsWith("darwin-")) await runAppleBuild(writeAppleConsumerReceipt(root, artifacts))
