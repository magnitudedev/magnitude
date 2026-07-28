import {
  access,
  copyFile,
  mkdir,
  mkdtemp,
  rename,
  rm,
  writeFile,
} from "node:fs/promises"
import { constants } from "node:fs"
import { basename, delimiter, resolve } from "node:path"
import { getDefaultBunTarget } from "../../scripts/release-target"
import { buildIcnBinary } from "./compile"

const PROJECT_ROOT = resolve(import.meta.dir, "../..")

export type LocalIcnBackend = "cpu" | "cuda" | "metal" | "vulkan"

const executableExists = async (name: string): Promise<boolean> => {
  const path = Bun.which(name)
  if (!path) return false
  return access(path, constants.X_OK).then(() => true, () => false)
}

const commandSucceeds = async (command: readonly string[]): Promise<boolean> => {
  const child = Bun.spawn([...command], { stdout: "ignore", stderr: "ignore" })
  return (await child.exited) === 0
}

const selectBackend = async (): Promise<LocalIcnBackend> => {
  const requested = process.env.MAGNITUDE_ICN_BACKEND?.trim().toLowerCase()
  if (requested && !["cpu", "cuda", "metal", "vulkan"].includes(requested)) {
    throw new Error("MAGNITUDE_ICN_BACKEND must be cpu, cuda, metal, or vulkan")
  }
  if (requested === "metal" && (process.platform !== "darwin" || process.arch !== "arm64")) {
    throw new Error("Metal development builds require Apple Silicon")
  }
  if (requested === "cuda" && !await executableExists("nvcc")) {
    throw new Error("CUDA development builds require nvcc")
  }
  if (requested) return requested as LocalIcnBackend
  if (process.platform === "darwin" && process.arch === "arm64") return "metal"
  if (
    process.platform === "linux" &&
    await executableExists("nvcc") &&
    await commandSucceeds(["nvidia-smi", "-L"])
  ) return "cuda"
  return "cpu"
}

const run = async (
  command: readonly string[],
  environment: Readonly<Record<string, string | undefined>> = process.env,
): Promise<void> => {
  const child = Bun.spawn([...command], {
    cwd: PROJECT_ROOT,
    env: environment,
    stdin: "ignore",
    stdout: "inherit",
    stderr: "inherit",
  })
  const code = await child.exited
  if (code !== 0) {
    throw new Error(`command failed with exit code ${code}: ${command.join(" ")}`)
  }
}

export const buildLocalIcn = async (): Promise<{
  readonly backend: LocalIcnBackend
  readonly installationPath: string
}> => {
  const backend = await selectBackend()
  await run(["bun", "run", "icn:catalog:hydrate"])
  const build = await buildIcnBinary({
    target: getDefaultBunTarget(),
    profile: `development-${backend}`,
    features: [
      "mtmd",
      "dynamic-backends",
      ...(backend === "cpu" ? [] : [backend === "cuda" ? "cuda" : backend]),
    ],
    release: false,
    clean: false,
  })
  const target = resolve(PROJECT_ROOT, "inference/target")
  const staging = await mkdtemp(resolve(target, ".development-"))
  const destination = resolve(target, "development")
  try {
    for (const directory of ["bin", "runtime", "backends", "catalog"]) {
      await mkdir(resolve(staging, directory), { recursive: true, mode: 0o700 })
    }
    const executable = process.platform === "win32"
      ? "magnitude-icn.exe"
      : "magnitude-icn"
    await copyFile(build.binary, resolve(staging, "bin", executable))
    for (const source of build.runtimeLibraries) {
      await copyFile(source, resolve(staging, "runtime", basename(source)))
    }
    const modules = build.backendModules.filter((source) => {
      const name = basename(source).toLowerCase()
      return name.includes("cpu") || (backend !== "cpu" && name.includes(backend))
    })
    if (
      !modules.some((source) => basename(source).toLowerCase().includes("cpu")) ||
      (backend !== "cpu" &&
        !modules.some((source) => basename(source).toLowerCase().includes(backend)))
    ) {
      throw new Error(`development build did not emit the ${backend} backend`)
    }
    for (const source of modules) {
      await copyFile(source, resolve(staging, "backends", basename(source)))
    }
    await copyFile(
      resolve(target, "catalog-inputs/release-catalog.lock.json"),
      resolve(staging, "catalog/release-catalog.lock.json"),
    )
    await copyFile(
      resolve(target, "catalog-inputs/model-planner-inputs.bundle"),
      resolve(staging, "catalog/model-planner-inputs.bundle"),
    )
    const installation = resolve(staging, "installation.json")
    await writeFile(installation, `${JSON.stringify({
      schemaVersion: 1,
      backend,
      nativeBuild: build.identity.native_build,
      backendModuleAbi: build.identity.backend_module_abi,
    }, null, 2)}\n`)
    const loader = process.platform === "win32"
      ? "PATH"
      : process.platform === "darwin"
        ? "DYLD_LIBRARY_PATH"
        : "LD_LIBRARY_PATH"
    await run([
      resolve(staging, "bin", executable),
      "installation-check",
      "--installation",
      installation,
    ], {
      ...process.env,
      [loader]: [resolve(staging, "runtime"), process.env[loader]]
        .filter(Boolean)
        .join(delimiter),
    })
    await rm(destination, { recursive: true, force: true })
    await rename(staging, destination)
    return {
      backend,
      installationPath: resolve(destination, "installation.json"),
    }
  } catch (cause) {
    await rm(staging, { recursive: true, force: true })
    throw cause
  }
}

if (import.meta.main) {
  const result = await buildLocalIcn()
  if (process.argv.includes("--serve")) {
    const executable = process.platform === "win32"
      ? "magnitude-icn.exe"
      : "magnitude-icn"
    const loader = process.platform === "win32"
      ? "PATH"
      : process.platform === "darwin"
        ? "DYLD_LIBRARY_PATH"
        : "LD_LIBRARY_PATH"
    const child = Bun.spawn([
      resolve(PROJECT_ROOT, "inference/target/development/bin", executable),
      "serve",
      "--installation",
      result.installationPath,
      ...process.argv.slice(2).filter((argument) => argument !== "--serve"),
    ], {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        [loader]: [
          resolve(PROJECT_ROOT, "inference/target/development/runtime"),
          process.env[loader],
        ].filter(Boolean).join(delimiter),
      },
      stdin: "inherit",
      stdout: "inherit",
      stderr: "inherit",
    })
    process.exit(await child.exited)
  }
  console.log(`ICN development installation ready at ${result.installationPath}`)
}
