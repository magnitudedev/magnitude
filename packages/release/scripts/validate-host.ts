import { writeAppleConsumerReceipt } from "./apple/distribution"
import { runAppleBuild } from "./apple/compile-bun"
import { fileSha256 } from "./build/common"
import { readFile } from "node:fs/promises"
import { resolve } from "node:path"
import { Schema } from "effect"
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

const artifacts = hostId.startsWith("darwin-") ? await Promise.all(["cli", "acn", "icn-base"].map(async (kind) => {
  const metadata = Schema.decodeUnknownSync(Schema.parseJson(ReleaseArtifactSchema))(await readFile(resolve(root, `${kind}-${hostId}.artifact.json`), "utf8"))
  if (await fileSha256(resolve(root, metadata.filename)) !== metadata.sha256) throw new Error(`Consumer received changed ${metadata.id} bytes`)
  return metadata
})) : []

await smokeHostArchives(
  host,
  resolve(root, cliArchive(hostId)),
  resolve(root, acnArchive(hostId)),
  resolve(root, icnBaseArchive(hostId)),
  artifact
)

if (hostId.startsWith("darwin-")) await runAppleBuild(writeAppleConsumerReceipt(root, artifacts))
