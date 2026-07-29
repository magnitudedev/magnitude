import { createHash } from "node:crypto"
import {
  copyFile,
  mkdir,
  readdir,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises"
import { basename, resolve } from "node:path"
import { Option, Schema } from "effect"
import {
  ReleaseArtifactSchema,
  ReleaseManifestSchema,
  type ReleaseArtifact,
} from "../src/contracts"
import {
  acnArchive,
  backendArchive,
  backendPacks,
  cliArchive,
  icnBaseArchive,
  releaseHosts,
} from "../src/targets"
import { fileSha256, run } from "./build/common"

const PROJECT_ROOT = resolve(import.meta.dir, "../../..")

const files = async (root: string): Promise<readonly string[]> => {
  const found: string[] = []
  const visit = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = resolve(directory, entry.name)
      if (entry.isDirectory()) await visit(path)
      else if (entry.isFile()) found.push(path)
    }
  }
  await visit(root)
  return found.sort()
}

const required = (name: string, fallback?: string): string => {
  const value = process.env[name]?.trim() || fallback
  if (!value) throw new Error(`${name} is required`)
  return value
}

const expectedArtifacts = new Map<string, string>([
  ...releaseHosts.flatMap((host) => [
    [`cli-${host.id}`, cliArchive(host.id)] as const,
    [`acn-${host.id}`, acnArchive(host.id)] as const,
    [`icn-base-${host.id}`, icnBaseArchive(host.id)] as const,
  ]),
  ...backendPacks.map((pack) =>
    [`icn-backend-${pack.id}`, backendArchive(pack)] as const
  ),
])

const archiveListing = async (archive: string): Promise<readonly string[]> =>
  (await run(["tar", "-tzf", archive]))
    .split("\n")
    .filter((entry) => entry.length > 0)
    .sort()

const validateLayout = async (
  artifact: ReleaseArtifact,
  archive: string,
): Promise<void> => {
  const listing = await archiveListing(archive)
  const host = Option.getOrThrow(artifact.host)
  const extension = host === "windows-x64-msvc" ? ".exe" : ""
  if (artifact.kind === "cli" || artifact.kind === "acn") {
    const expected = [`bin/magnitude-${artifact.kind}${extension}`]
    if (JSON.stringify(listing) !== JSON.stringify(expected)) {
      throw new Error(`${artifact.id} has an invalid executable archive layout`)
    }
    return
  }
  if (listing.some((entry) =>
    entry.startsWith("/") ||
    entry.includes("\\") ||
    entry.split("/").some((part) => part === "" || part === "." || part === "..")
  )) {
    throw new Error(`${artifact.id} contains an unsafe archive path`)
  }
  if (artifact.kind === "icn-base") {
    for (const requiredPath of [
      `bin/magnitude-icn${extension}`,
      "catalog/release-catalog.lock.json",
      "catalog/model-planner-inputs.bundle",
    ]) {
      if (!listing.includes(requiredPath)) {
        throw new Error(`${artifact.id} is missing ${requiredPath}`)
      }
    }
    const backendNames = listing
      .filter((entry) => entry.startsWith("backends/"))
      .map((entry) => basename(entry).toLowerCase())
    if (
      !backendNames.some((name) => name.includes("cpu")) ||
      backendNames.some((name) =>
        name.includes("metal") || name.includes("cuda") || name.includes("vulkan")
      )
    ) {
      throw new Error(`${artifact.id} does not contain exactly the CPU backend family`)
    }
    if (listing.some((entry) =>
      !entry.startsWith("bin/") &&
      !entry.startsWith("catalog/") &&
      !entry.startsWith("runtime/") &&
      !entry.startsWith("backends/")
    )) {
      throw new Error(`${artifact.id} contains an unexpected path`)
    }
    return
  }
  const backend = Option.getOrThrow(artifact.backend)
  const expectedModule = backendPacks.find(
    (pack) => `icn-backend-${pack.id}` === artifact.id,
  )?.module
  if (
    !expectedModule ||
    !listing.includes(`backends/${expectedModule}`) ||
    listing.filter((entry) => entry.startsWith("backends/")).length !== 1 ||
    listing.some((entry) =>
      !entry.startsWith("runtime/") && !entry.startsWith("backends/")
    ) ||
    backend === "cpu"
  ) {
    throw new Error(`${artifact.id} has an invalid backend-pack layout`)
  }
}

const archiveEntry = async (
  archive: string,
  entry: string,
): Promise<Buffer> => {
  const child = Bun.spawn(["tar", "-xOf", archive, entry], {
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  })
  const [code, bytes, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).arrayBuffer().then((value) => Buffer.from(value)),
    new Response(child.stderr).text(),
  ])
  if (code !== 0) {
    throw new Error(`unable to read ${entry} from ${basename(archive)}: ${stderr}`)
  }
  return bytes
}

const input = resolve(process.argv[2] ?? "release-artifacts")
const output = resolve(process.argv[3] ?? "release-candidate")
const allFiles = await files(input)
const descriptorFiles = allFiles.filter((file) => file.endsWith(".artifact.json"))
const artifacts = await Promise.all(descriptorFiles.map(async (file) =>
  Schema.decodeUnknownSync(Schema.parseJson(ReleaseArtifactSchema))(
    await readFile(file, "utf8"),
  )
))
if (artifacts.length !== expectedArtifacts.size) {
  throw new Error(
    `candidate has ${artifacts.length} artifacts; expected ${expectedArtifacts.size}`,
  )
}
const byId = new Map(artifacts.map((artifact) => [artifact.id, artifact]))
if (byId.size !== artifacts.length) throw new Error("candidate artifact IDs are not unique")

const archiveById = new Map<string, string>()
for (const [id, filename] of expectedArtifacts) {
  const artifact = byId.get(id)
  if (!artifact || artifact.filename !== filename) {
    throw new Error(`${id} is missing or has the wrong filename`)
  }
  const matches = allFiles.filter((file) => basename(file) === filename)
  if (matches.length !== 1) {
    throw new Error(`${id} has ${matches.length} matching files`)
  }
  const archive = matches[0]!
  const info = await stat(archive)
  if (
    Number(info.size) !== artifact.bytes ||
    await fileSha256(archive) !== artifact.sha256
  ) {
    throw new Error(`${id} bytes differ from its descriptor`)
  }
  await validateLayout(artifact, archive)
  archiveById.set(id, archive)
}

let catalogLock: Buffer | undefined
let plannerBundleDigest: string | undefined
for (const host of releaseHosts) {
  const base = byId.get(`icn-base-${host.id}`)!
  const archive = archiveById.get(base.id)!
  const lock = await archiveEntry(archive, "catalog/release-catalog.lock.json")
  const bundle = await archiveEntry(archive, "catalog/model-planner-inputs.bundle")
  const bundleDigest = createHash("sha256").update(bundle).digest("hex")
  if (catalogLock && !catalogLock.equals(lock)) {
    throw new Error(`${base.id} contains a different catalog lock`)
  }
  if (plannerBundleDigest && plannerBundleDigest !== bundleDigest) {
    throw new Error(`${base.id} contains a different planner bundle`)
  }
  catalogLock = lock
  plannerBundleDigest = bundleDigest
}
for (const pack of backendPacks) {
  const artifact = byId.get(`icn-backend-${pack.id}`)!
  const base = byId.get(`icn-base-${pack.host}`)!
  if (
    Option.getOrThrow(artifact.requiredBaseId) !== base.id ||
    Option.getOrThrow(artifact.nativeBuild) !== Option.getOrThrow(base.nativeBuild) ||
    Option.getOrThrow(artifact.backendModuleAbi) !==
      Option.getOrThrow(base.backendModuleAbi)
  ) {
    throw new Error(`${artifact.id} is incompatible with ${base.id}`)
  }
}

const packageJson = JSON.parse(
  await readFile(resolve(PROJECT_ROOT, "packages/cli/package.json"), "utf8"),
) as { readonly version?: string }
const version = required("MAGNITUDE_RELEASE_VERSION", packageJson.version)
if (packageJson.version !== version) {
  throw new Error("package version differs from the release version")
}
const sourceCommit = required("MAGNITUDE_SOURCE_COMMIT")
if (!/^[a-f0-9]{40}$/.test(sourceCommit)) {
  throw new Error("MAGNITUDE_SOURCE_COMMIT must be a full lowercase commit SHA")
}
const manifest = Schema.decodeUnknownSync(ReleaseManifestSchema)({
  schemaVersion: 1,
  version,
  tag: `@magnitudedev/cli@${version}`,
  sourceCommit,
  artifacts: artifacts
    .slice()
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((artifact) => Schema.encodeSync(ReleaseArtifactSchema)(artifact)),
})
const manifestBytes = new TextEncoder().encode(
  `${JSON.stringify(Schema.encodeSync(ReleaseManifestSchema)(manifest), null, 2)}\n`,
)

await rm(output, { recursive: true, force: true })
await mkdir(output, { recursive: true, mode: 0o700 })
await writeFile(resolve(output, "magnitude-release.json"), manifestBytes)
for (const artifact of artifacts) {
  await copyFile(archiveById.get(artifact.id)!, resolve(output, artifact.filename))
}
