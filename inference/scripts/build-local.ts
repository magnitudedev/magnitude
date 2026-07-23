import { access } from "node:fs/promises"
import { constants } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..")

export type LocalIcnBackend = "cpu" | "cuda" | "metal"

interface LocalBackendEnvironment {
  readonly platform: NodeJS.Platform
  readonly arch: string
  readonly requested: string | undefined
  readonly nvidiaDriverAvailable: boolean
  readonly cudaToolkitAvailable: boolean
}

const commandSucceeds = async (command: string[]): Promise<boolean> => {
  const child = Bun.spawn(command, {
    stdout: "ignore",
    stderr: "ignore",
  })
  return (await child.exited) === 0
}

const executableExists = async (name: string): Promise<boolean> => {
  const path = Bun.which(name)
  if (!path) return false
  return access(path, constants.X_OK).then(() => true, () => false)
}

const detectEnvironment = async (): Promise<LocalBackendEnvironment> => {
  const nvidiaSmi = await executableExists("nvidia-smi")
  return {
    platform: process.platform,
    arch: process.arch,
    requested: process.env.MAGNITUDE_ICN_BACKEND?.trim().toLowerCase(),
    nvidiaDriverAvailable: nvidiaSmi
      && await commandSucceeds(["nvidia-smi", "-L"]),
    cudaToolkitAvailable: await executableExists("nvcc"),
  }
}

export const selectLocalIcnBackend = (
  environment: LocalBackendEnvironment,
): LocalIcnBackend => {
  const requested = environment.requested || "auto"
  if (!["auto", "cpu", "cuda", "metal"].includes(requested)) {
    throw new Error(
      `Unsupported MAGNITUDE_ICN_BACKEND=${requested}; expected auto, cpu, cuda, or metal`,
    )
  }

  if (requested === "metal") {
    if (environment.platform !== "darwin" || environment.arch !== "arm64") {
      throw new Error("The Metal ICN backend requires Apple Silicon")
    }
    return "metal"
  }

  const wantsCuda = requested === "cuda"
    || (requested === "auto"
      && environment.platform === "linux"
      && environment.nvidiaDriverAvailable)
  if (wantsCuda) {
    if (environment.platform !== "linux") {
      throw new Error("The local CUDA ICN build is supported only on Linux")
    }
    if (!environment.nvidiaDriverAvailable) {
      throw new Error("CUDA was requested, but no NVIDIA GPU is visible through nvidia-smi")
    }
    if (!environment.cudaToolkitAvailable) {
      throw new Error(
        "An NVIDIA GPU is available, but nvcc is missing. Install the CUDA toolkit or set MAGNITUDE_ICN_BACKEND=cpu.",
      )
    }
    return "cuda"
  }

  if (environment.platform === "darwin" && environment.arch === "arm64") {
    return "metal"
  }
  return "cpu"
}

const executableName = (): string =>
  process.platform === "win32" ? "magnitude-icn.exe" : "magnitude-icn"

export const localIcnBinaryPath = (backend: LocalIcnBackend): string =>
  resolve(PROJECT_ROOT, "inference/target", `dev-${backend}`, "debug", executableName())

export const buildLocalIcn = async (): Promise<{
  readonly backend: LocalIcnBackend
  readonly binaryPath: string
}> => {
  const backend = selectLocalIcnBackend(await detectEnvironment())
  const targetDir = resolve(PROJECT_ROOT, "inference/target", `dev-${backend}`)
  const command = [
    "cargo",
    "build",
    "--manifest-path",
    resolve(PROJECT_ROOT, "inference/Cargo.toml"),
    "-p",
    "icn-server",
    ...(backend === "cuda" ? ["--features", "cuda"] : []),
  ]
  console.log(`[build:icn] Building ${backend} development binary`)
  const child = Bun.spawn(command, {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      CARGO_TARGET_DIR: targetDir,
    },
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  })
  const exitCode = await child.exited
  if (exitCode !== 0) {
    throw new Error(`[build:icn] ${backend} build failed with exit code ${exitCode}`)
  }
  const binaryPath = localIcnBinaryPath(backend)
  await access(binaryPath, constants.X_OK)
  return { backend, binaryPath }
}

if (import.meta.main) {
  const result = await buildLocalIcn()
  console.log(`[build:icn] Ready: ${result.binaryPath}`)
}
