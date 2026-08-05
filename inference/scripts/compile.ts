import {
  readdir,
  rm,
  stat,
} from "node:fs/promises"
import { basename, delimiter, dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { IcnBinaryIdentity } from "@magnitudedev/icn-protocol"
import { Schema } from "effect"
import { getTargetInfo } from "../../scripts/release-target"

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..")
const CARGO_MANIFEST = resolve(PROJECT_ROOT, "inference/Cargo.toml")

const run = async (
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
    const diagnostics = [stderr, stdout]
      .filter((value) => value.trim().length > 0)
      .join("\n")
      .trim()
    throw new Error(
      `${command[0]} failed with exit ${code}: ${diagnostics}`,
    )
  }
  return stdout
}

export interface IcnBuild {
  readonly binary: string
  readonly identity: IcnBinaryIdentity
  readonly backendModules: readonly string[]
  readonly runtimeLibraries: readonly string[]
}

interface CargoMetadata {
  readonly packages: readonly {
    readonly id: string
    readonly name: string
  }[]
}

interface CargoMessage {
  readonly reason?: string
  readonly package_id?: string
  readonly out_dir?: string
  readonly executable?: string
  readonly target?: { readonly name?: string }
  readonly message?: { readonly rendered?: string | null }
}

export const readCargoMessages = async (
  stream: ReadableStream<Uint8Array>,
  writeDiagnostic: (rendered: string) => void = (rendered) =>
    process.stderr.write(rendered),
): Promise<readonly CargoMessage[]> => {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  const messages: CargoMessage[] = []
  let pending = ""

  const accept = (line: string): void => {
    if (line.trim().length === 0) return
    const message = JSON.parse(line) as CargoMessage
    if (
      message.reason === "compiler-message" &&
      typeof message.message?.rendered === "string"
    ) writeDiagnostic(message.message.rendered)
    if (
      message.reason === "build-script-executed" ||
      message.reason === "compiler-artifact"
    ) messages.push(message)
  }

  try {
    while (true) {
      const next = await reader.read()
      if (next.done) break
      const lines = `${pending}${decoder.decode(next.value, { stream: true })}`
        .split("\n")
      pending = lines.pop() ?? ""
      for (const line of lines) accept(line)
    }
    pending += decoder.decode()
    accept(pending)
    return messages
  } finally {
    reader.releaseLock()
  }
}

const runCargoBuild = async (
  command: readonly string[],
  options: {
    readonly cwd: string
    readonly env: Readonly<Record<string, string | undefined>>
  },
): Promise<readonly CargoMessage[]> => {
  const child = Bun.spawn([...command], {
    cwd: options.cwd,
    env: options.env,
    stdin: "ignore",
    stdout: "pipe",
    stderr: "inherit",
  })
  const [code, messages] = await Promise.all([
    child.exited,
    readCargoMessages(child.stdout),
  ])
  if (code !== 0) {
    throw new Error(`${command[0]} failed with exit ${code}`)
  }
  return messages
}

const rustTarget = (target: string): string => {
  const { platform, arch } = getTargetInfo(target)
  const mapped: Record<string, string> = {
    "darwin-arm64": "aarch64-apple-darwin",
    "darwin-x64": "x86_64-apple-darwin",
    "linux-arm64": "aarch64-unknown-linux-gnu",
    "linux-x64": "x86_64-unknown-linux-gnu",
    "windows-x64": "x86_64-pc-windows-msvc",
  }
  const value = mapped[`${platform}-${arch}`]
  if (!value) throw new Error(`No ICN Rust target for ${target}`)
  return value
}

const nativeRuntimeLinkageEnvironment = (
  target: string,
): Readonly<Record<string, string>> => {
  const { platform } = getTargetInfo(target)
  if (platform === "linux") {
    return {
      CMAKE_BUILD_RPATH_USE_ORIGIN: "ON",
      CMAKE_INSTALL_RPATH: "$ORIGIN;$ORIGIN/../runtime",
    }
  }
  if (platform === "darwin") {
    return {
      CMAKE_INSTALL_RPATH: "@loader_path;@loader_path/../runtime",
    }
  }
  return {}
}

const filesIn = async (directory: string): Promise<readonly string[]> => {
  try {
    const entries = await readdir(directory, { withFileTypes: true })
    return entries
      .filter((entry) => entry.isFile() || entry.isSymbolicLink())
      .map((entry) => resolve(directory, entry.name))
      .sort()
  } catch (cause) {
    if (
      cause instanceof Error &&
      "code" in cause &&
      cause.code === "ENOENT"
    ) return []
    throw cause
  }
}

const isRuntimeLibrary = (file: string): boolean => {
  const name = basename(file)
  return name.endsWith(".dylib") ||
    name.endsWith(".dll") ||
    name.includes(".so")
}

const isBackendModule = (file: string): boolean => {
  const name = basename(file).toLowerCase()
  return [
    "libggml-cpu",
    "libggml-metal",
    "libggml-cuda",
    "libggml-vulkan",
    "ggml-cpu",
    "ggml-metal",
    "ggml-cuda",
    "ggml-vulkan",
  ].some((prefix) => name.startsWith(prefix))
}

const readIdentity = async (
  binary: string,
  runtimeDirectories: readonly string[],
): Promise<IcnBinaryIdentity> => {
  const loader = process.platform === "win32"
    ? "PATH"
    : process.platform === "darwin"
      ? "DYLD_LIBRARY_PATH"
      : "LD_LIBRARY_PATH"
  const stdout = await run([binary, "version", "--json"], {
    env: {
      ...process.env,
      [loader]: [...runtimeDirectories, process.env[loader]]
        .filter(Boolean)
        .join(delimiter),
    },
  })
  const value = Schema.decodeUnknownSync(
    Schema.parseJson(IcnBinaryIdentity),
  )(stdout)
  if (value.api_version !== 1) {
    throw new Error("ICN identity probe returned an invalid contract")
  }
  return value
}

export interface BuildIcnInput {
  readonly target: string
  readonly profile: string
  readonly features: readonly string[]
  readonly release?: boolean
  readonly clean?: boolean
  readonly buildEnvironment?: Readonly<Record<string, string>>
}

export const buildIcnBinary = async ({
  target,
  profile,
  features,
  release = true,
  clean = true,
  buildEnvironment = {},
}: BuildIcnInput): Promise<IcnBuild> => {
  const cargoTarget = rustTarget(target)
  const targetDirectory = resolve(
    PROJECT_ROOT,
    "inference/target",
    `release-${profile}`,
  )
  if (clean) await rm(targetDirectory, { recursive: true, force: true })

  const metadata = JSON.parse(await run([
    "cargo",
    "metadata",
    "--format-version",
    "1",
    "--manifest-path",
    CARGO_MANIFEST,
  ], { cwd: PROJECT_ROOT })) as CargoMetadata
  const nativePackage = metadata.packages.find(
    (candidate) => candidate.name === "llama-cpp-sys-2",
  )
  if (!nativePackage) {
    throw new Error("Cargo metadata has no llama-cpp-sys-2 package")
  }

  const messages = await runCargoBuild([
    "cargo",
    "build",
    ...(release ? ["--release"] : []),
    "--manifest-path",
    CARGO_MANIFEST,
    "-p",
    "icn-server",
    "--target",
    cargoTarget,
    "--no-default-features",
    "--features",
    [...new Set(features)].join(","),
    "--message-format",
    "json-render-diagnostics",
  ], {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      ...buildEnvironment,
      ...nativeRuntimeLinkageEnvironment(target),
      CARGO_TARGET_DIR: targetDirectory,
    },
  })
  const outDirectories = messages
    .filter((message) =>
      message.reason === "build-script-executed" &&
      message.package_id === nativePackage.id &&
      typeof message.out_dir === "string"
    )
    .map((message) => message.out_dir!)
  if (outDirectories.length !== 1) {
    throw new Error(`Cargo reported ${outDirectories.length} native output directories`)
  }
  const binaryMessages = messages.filter((message) =>
    message.reason === "compiler-artifact" &&
    message.target?.name === "magnitude-icn" &&
    typeof message.executable === "string"
  )
  if (binaryMessages.length !== 1) {
    throw new Error(`Cargo reported ${binaryMessages.length} ICN executables`)
  }
  const binary = binaryMessages[0]!.executable!
  const nativeOutput = outDirectories[0]!
  const backendModules = await filesIn(resolve(nativeOutput, "backends"))
  if (backendModules.length === 0) {
    throw new Error("ICN build emitted no dynamic backend modules")
  }
  const installedRuntimeLibraries = (
    await filesIn(resolve(nativeOutput, "lib"))
  ).filter((file) => isRuntimeLibrary(file) && !isBackendModule(file))
  const installedNames = new Set(
    installedRuntimeLibraries.map((file) => basename(file)),
  )
  const supplementalRuntimeLibraries = (
    await filesIn(resolve(nativeOutput, "build", "bin"))
  ).filter((file) =>
    isRuntimeLibrary(file) &&
    !isBackendModule(file) &&
    !installedNames.has(basename(file))
  )
  const runtimeLibraries = [
    ...installedRuntimeLibraries,
    ...supplementalRuntimeLibraries,
  ]
  const identity = await readIdentity(
    binary,
    [...new Set(runtimeLibraries.map(dirname))],
  )
  for (const file of [binary, ...backendModules, ...runtimeLibraries]) {
    if (!(await stat(file)).isFile()) throw new Error(`missing ICN output ${file}`)
  }
  return { binary, identity, backendModules, runtimeLibraries }
}
