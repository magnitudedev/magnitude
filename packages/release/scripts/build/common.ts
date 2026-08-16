import { createHash } from "node:crypto"
import { createReadStream, createWriteStream } from "node:fs"
import {
  mkdir,
  rm,
  stat,
  writeFile,
} from "node:fs/promises"
import { basename, dirname } from "node:path"
import { pipeline } from "node:stream/promises"
import { createGzip } from "node:zlib"
import { Schema } from "effect"
import { pack } from "tar-stream"
import {
  ReleaseArtifactSchema,
  type ReleaseArtifact,
} from "../../src/contracts"
import {
  MACOS_DEPLOYMENT_TARGET,
  type HostId,
} from "../../src/targets"

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

const compareVersions = (left: string, right: string): number => {
  const leftParts = left.split(".").map(Number)
  const rightParts = right.split(".").map(Number)
  for (let index = 0; index < Math.max(leftParts.length, rightParts.length); index += 1) {
    const difference = (leftParts[index] ?? 0) - (rightParts[index] ?? 0)
    if (difference !== 0) return difference
  }
  return 0
}

export const verifyAppleDeploymentTarget = async (
  host: HostId,
  files: readonly string[],
): Promise<void> => {
  if (!host.startsWith("darwin-")) return
  const architecture = host === "darwin-arm64" ? "arm64" : "x86_64"
  for (const file of files) {
    const report = await run([
      "vtool",
      "-arch",
      architecture,
      "-show-build",
      file,
    ])
    const platform = report.match(/^\s*platform\s+(\S+)\s*$/m)?.[1]
    const minimum = report.match(/^\s*minos\s+(\d+(?:\.\d+){1,2})\s*$/m)?.[1]
    if (platform !== "MACOS" || minimum === undefined) {
      throw new Error(`${basename(file)} has no macOS deployment target`)
    }
    if (compareVersions(minimum, MACOS_DEPLOYMENT_TARGET) > 0) {
      throw new Error(
        `${basename(file)} requires macOS ${minimum}; release floor is macOS ${MACOS_DEPLOYMENT_TARGET}`,
      )
    }
  }
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
  const orderedSources = sources
    .map((source, index) => ({ source, path: paths[index]! }))
    .sort((left, right) => left.path.localeCompare(right.path))
  await mkdir(dirname(archive), { recursive: true, mode: 0o700 })
  const tar = pack()
  const writeArchive = pipeline(
    tar,
    createGzip(),
    createWriteStream(archive, { mode: 0o600 }),
  )
  try {
    for (const { source, path } of orderedSources) {
      const sourceInfo = await stat(source.source)
      if (!sourceInfo.isFile()) {
        throw new Error(`${draft.id} archive source ${source.source} is not a file`)
      }
      const entry = tar.entry({
        name: path,
        type: "file",
        size: sourceInfo.size,
        mode: source.mode,
        mtime: new Date(0),
        uid: 0,
        gid: 0,
      })
      await pipeline(createReadStream(source.source), entry)
    }
    tar.finalize()
    await writeArchive
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
  } catch (cause) {
    tar.destroy(cause instanceof Error ? cause : new Error(String(cause)))
    await writeArchive.catch(() => undefined)
    await rm(archive, { force: true })
    throw cause
  }
}
