import { resolve } from "path"
import { randomUUID } from "node:crypto"
import { link, mkdir, readFile, unlink, writeFile } from "node:fs/promises"

const PROJECT_ROOT = resolve(import.meta.dir, "..", "..", "..")
const SOURCE_PACKAGE_JSON = resolve(PROJECT_ROOT, "packages/cli/package.json")
const REVISION_FILE = resolve(PROJECT_ROOT, "packages/version/acn-revision.json")
const OUTPUT_FILE = resolve(PROJECT_ROOT, "packages/version/src/version.generated.ts")
const OUTPUT_RELATIVE = "packages/version/src/version.generated.ts"

const isDev = process.argv.includes("--dev")
const ACN_REVISION_BAND_SIZE = 1_000_000

interface PublishedRevisionRecord {
  readonly version: string
  readonly revision: number
}

const readGitShortHash = (): string => {
  const proc = Bun.spawnSync(["git", "rev-parse", "--short", "HEAD"], {
    cwd: PROJECT_ROOT,
    stdout: "pipe",
    stderr: "ignore",
  })
  if (proc.exitCode !== 0) return "unknown"
  return proc.stdout.toString().trim() || "unknown"
}

const gitOutput = (args: ReadonlyArray<string>): Uint8Array => {
  const proc = Bun.spawnSync(["git", ...args], {
    cwd: PROJECT_ROOT,
    stdout: "pipe",
    stderr: "ignore",
  })
  return proc.exitCode === 0 ? proc.stdout : new Uint8Array()
}

const developmentFingerprint = async (): Promise<string> => {
  const hash = new Bun.CryptoHasher("sha256")
  hash.update(readGitShortHash())
  hash.update(gitOutput([
    "diff",
    "--no-ext-diff",
    "--binary",
    "HEAD",
    "--",
    ".",
    `:(exclude)${OUTPUT_RELATIVE}`,
  ]))
  const untracked = new TextDecoder().decode(gitOutput([
    "ls-files",
    "--others",
    "--exclude-standard",
    "-z",
  ])).split("\0").filter((path) => path.length > 0 && path !== OUTPUT_RELATIVE).sort()
  for (const relative of untracked) {
    hash.update(relative)
    hash.update(await Bun.file(resolve(PROJECT_ROOT, relative)).arrayBuffer())
  }
  return hash.digest("hex").slice(0, 16)
}

const developmentVersion = async (
  baseVersion: string,
  fingerprint: string,
): Promise<string> => {
  const gitPath = new TextDecoder().decode(gitOutput([
    "rev-parse",
    "--git-path",
    "magnitude-dev-versions",
  ])).trim()
  const cacheDirectory = resolve(PROJECT_ROOT, gitPath || ".git/magnitude-dev-versions")
  const cachePath = resolve(cacheDirectory, fingerprint)
  const candidate = `${baseVersion}+dev.${readGitShortHash()}.${Date.now()}`
  const temporary = `${cachePath}.${process.pid}.${Date.now()}.tmp`
  await mkdir(cacheDirectory, { recursive: true })
  await writeFile(temporary, `${candidate}\n`, { flag: "wx", mode: 0o600 })
  try {
    await link(temporary, cachePath)
    return candidate
  } catch (error) {
    if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error
    return (await readFile(cachePath, "utf8")).trim()
  } finally {
    await unlink(temporary).catch(() => undefined)
  }
}

const gitPrivateDirectory = (name: string): string => {
  const value = new TextDecoder().decode(gitOutput([
    "rev-parse",
    "--git-path",
    name,
  ])).trim()
  return resolve(PROJECT_ROOT, value || `.git/${name}`)
}

const readPublishedRevision = async (): Promise<PublishedRevisionRecord> => {
  const value = JSON.parse(await readFile(REVISION_FILE, "utf8")) as Partial<PublishedRevisionRecord>
  if (
    typeof value.version !== "string" ||
    !Number.isSafeInteger(value.revision) ||
    value.revision! <= 0 ||
    value.revision! % ACN_REVISION_BAND_SIZE !== 0
  ) {
    throw new Error("packages/version/acn-revision.json is malformed")
  }
  return { version: value.version, revision: value.revision! }
}

const developmentRevision = async (
  published: number,
  fingerprint: string,
): Promise<number> => {
  const root = gitPrivateDirectory("magnitude-dev-revisions")
  const claims = resolve(root, "claims")
  const fingerprints = resolve(root, "fingerprints")
  const fingerprintPath = resolve(fingerprints, fingerprint)
  await mkdir(claims, { recursive: true })
  await mkdir(fingerprints, { recursive: true })

  const readExisting = async (): Promise<number | undefined> => {
    try {
      const revision = Number((await readFile(fingerprintPath, "utf8")).trim())
      if (
        Number.isSafeInteger(revision) &&
        revision > published &&
        revision < published + ACN_REVISION_BAND_SIZE
      ) return revision
      throw new Error(`Development revision for ${fingerprint} is outside the current band`)
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return undefined
      throw error
    }
  }

  const existing = await readExisting()
  if (existing !== undefined) return existing

  for (let overlay = 1; overlay < ACN_REVISION_BAND_SIZE; overlay += 1) {
    const revision = published + overlay
    const temporary = resolve(root, `.claim-${process.pid}-${randomUUID()}`)
    await writeFile(temporary, `${revision}\n`, { flag: "wx", mode: 0o600 })
    try {
      try {
        await link(temporary, resolve(claims, String(revision)))
      } catch (error) {
        if (error instanceof Error && "code" in error && error.code === "EEXIST") continue
        throw error
      }
      try {
        await link(temporary, fingerprintPath)
        return revision
      } catch (error) {
        if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error
        const raced = await readExisting()
        if (raced !== undefined) return raced
      }
    } finally {
      await unlink(temporary).catch(() => undefined)
    }
  }
  throw new Error("The current ACN development revision band is exhausted")
}

async function main() {
  const packageJson = JSON.parse(await Bun.file(SOURCE_PACKAGE_JSON).text()) as { version?: string }

  if (!packageJson.version) {
    throw new Error(`No version found in ${SOURCE_PACKAGE_JSON}`)
  }

  const published = await readPublishedRevision()
  if (published.version !== packageJson.version) {
    throw new Error(
      `ACN published revision is for ${published.version}; expected ${packageJson.version}`,
    )
  }

  const fingerprint = isDev ? await developmentFingerprint() : undefined
  const version = isDev
    ? await developmentVersion(packageJson.version, fingerprint!)
    : packageJson.version
  const revision = fingerprint === undefined
    ? published.revision
    : await developmentRevision(published.revision, fingerprint)

  const contents =
    `// Generated by packages/version/scripts/generate-version.ts\n` +
    `// Source: packages/cli/package.json\n` +
    (fingerprint === undefined ? "" : `// Development fingerprint: ${fingerprint}\n`) +
    `export const MAGNITUDE_VERSION = ${JSON.stringify(version)}\n` +
    `export const ACN_COORDINATION_REVISION = ${revision}\n`

  await Bun.write(OUTPUT_FILE, contents)
}

await main()
