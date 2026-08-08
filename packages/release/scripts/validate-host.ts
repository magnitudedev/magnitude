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

await smokeHostArchives(
  host,
  resolve(root, cliArchive(hostId)),
  resolve(root, acnArchive(hostId)),
  resolve(root, icnBaseArchive(hostId)),
  artifact
)
