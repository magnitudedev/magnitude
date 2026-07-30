import { createHash } from "node:crypto"
import {
  chmod,
  copyFile,
  mkdir,
  mkdtemp,
  rm,
  stat,
  writeFile,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import { basename, dirname, resolve } from "node:path"
import { Schema } from "effect"
import {
  ReleaseArtifactSchema,
  type ReleaseArtifact,
} from "../../src/contracts"

export interface ArchiveSource {
  readonly path: string
  readonly source: string
  readonly mode: number
}

const normalizedArchivePath = (value: string): string => {
  const segments = value.split("/")
  if (
    value.length === 0 ||
    value.includes("\\") ||
    value.startsWith("/") ||
    segments.some((segment) => segment === "" || segment === "." || segment === "..")
  ) {
    throw new Error(`invalid release archive path ${value}`)
  }
  return value
}

export const fileSha256 = async (file: string): Promise<string> => {
  const hash = createHash("sha256")
  for await (const chunk of Bun.file(file).stream()) hash.update(chunk)
  return hash.digest("hex")
}

export const run = async (
  command: readonly string[],
  options: {
    readonly cwd?: string
    readonly env?: Readonly<Record<string, string | undefined>>
  } = {},
): Promise<string> => {
  const child = Bun.spawn([...command], {
    cwd: options.cwd,
    env: options.env,
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  })
  const [code, stdout, stderr] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
  ])
  if (code !== 0) {
    throw new Error(
      `${command[0]} failed with exit ${code}: ${(stderr || stdout).trim().slice(0, 4_000)}`,
    )
  }
  return stdout
}

export const buildArchive = async (
  archive: string,
  descriptor: string,
  draft: Omit<ReleaseArtifact, "filename" | "bytes" | "sha256">,
  sources: readonly ArchiveSource[],
): Promise<ReleaseArtifact> => {
  const paths = sources.map((source) => normalizedArchivePath(source.path))
  if (paths.length === 0 || new Set(paths).size !== paths.length) {
    throw new Error(`${draft.id} contains duplicate or no archive files`)
  }
  const staging = await mkdtemp(resolve(tmpdir(), "magnitude-release-"))
  try {
    await Promise.all(sources.map(async (source) => {
      const destination = resolve(staging, source.path)
      await mkdir(dirname(destination), { recursive: true, mode: 0o700 })
      await copyFile(source.source, destination)
      await chmod(destination, source.mode)
    }))
    await mkdir(dirname(archive), { recursive: true, mode: 0o700 })
    await run([
      "tar",
      "-czf",
      archive,
      "-C",
      staging,
      ...paths.slice().sort(),
    ], {
      env: {
        ...process.env,
        // Prevent macOS tar from adding AppleDouble `._*` metadata entries,
        // which would violate the release archive's exact file layout.
        COPYFILE_DISABLE: "1",
      },
    })
    const info = await stat(archive)
    const artifact = Schema.validateSync(ReleaseArtifactSchema)({
      ...draft,
      filename: basename(archive),
      bytes: Number(info.size),
      sha256: await fileSha256(archive),
    })
    await writeFile(
      descriptor,
      `${JSON.stringify(Schema.encodeSync(ReleaseArtifactSchema)(artifact), null, 2)}\n`,
      { flag: "wx", mode: 0o600 },
    )
    return artifact
  } finally {
    await rm(staging, { recursive: true, force: true })
  }
}
