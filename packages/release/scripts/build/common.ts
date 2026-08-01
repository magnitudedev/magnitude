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
import type { HostId } from "../../src/targets"

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

export interface OwnedLoaderPathInputs {
  readonly host: HostId
  readonly executable?: string
  readonly modules: readonly string[]
  readonly runtime: readonly string[]
}

export const verifyOwnedLoaderPaths = async ({
  host,
  executable,
  modules,
  runtime,
}: OwnedLoaderPathInputs): Promise<void> => {
  if (host.startsWith("windows-")) return

  const inspect = async (file: string): Promise<readonly string[]> => {
    if (host.startsWith("linux-")) {
      const dynamic = await run(["readelf", "-d", file])
      return [...dynamic.matchAll(
        /\((?:RUNPATH|RPATH)\).*\[([^\]]+)\]/g,
      )].flatMap((match) => match[1]!.split(":"))
    }
    const commands = await run(["otool", "-l", file])
    return [...commands.matchAll(
      /LC_RPATH[\s\S]*?\n\s*path ([^ ]+) \(offset/g,
    )].map((match) => match[1]!)
  }

  const expectedExecutable = host.startsWith("linux-")
    ? ["$ORIGIN/../runtime"]
    : ["@loader_path/../runtime"]
  const expectedLibrary = host.startsWith("linux-")
    ? ["$ORIGIN", "$ORIGIN/../runtime"]
    : ["@loader_path", "@loader_path/../runtime"]
  const verify = async (
    file: string,
    allowed: readonly string[],
    exact: boolean,
  ): Promise<void> => {
    const actual = await inspect(file)
    if (
      actual.some((path) => !allowed.includes(path)) ||
      (exact && (
        actual.length !== allowed.length ||
        allowed.some((path) => !actual.includes(path))
      ))
    ) {
      throw new Error(
        `${basename(file)} has loader paths ${JSON.stringify(actual)}; allowed ${JSON.stringify(allowed)}`,
      )
    }
  }

  if (executable !== undefined) {
    await verify(executable, expectedExecutable, true)
  }
  await Promise.all(
    modules.map((module) => verify(module, expectedLibrary, true)),
  )
  // Vendor runtime libraries may need no rpath. If present, it may only
  // address this installation's runtime directory.
  await Promise.all(
    runtime.map((library) => verify(library, expectedLibrary, false)),
  )
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
