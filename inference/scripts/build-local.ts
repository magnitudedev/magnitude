import { copyFile, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises"
import { basename, resolve } from "node:path"
import { IcnInstallationDeclaration } from "@magnitudedev/icn-protocol"
import { INFERENCE_PLANNER_BUNDLE, inferenceExecutablePath } from "@magnitudedev/release"
import { currentHost, hostById } from "@magnitudedev/release/targets"
import { Schema } from "effect"
import { buildInferenceBinary, INFERENCE_ROOT } from "./compile"
import { installationEnvironment } from "./smoke"

/** The development installation, laid out exactly as a release installation. */
export const DEVELOPMENT_INSTALLATION = resolve(INFERENCE_ROOT, "target/development")
const CATALOG_INPUTS = resolve(INFERENCE_ROOT, "target/catalog-inputs")

const run = async (command: readonly string[], diagnostics: "all" | "errors"): Promise<void> => {
  const child = Bun.spawn([...command], {
    cwd: INFERENCE_ROOT,
    stdin: "ignore",
    stdout: diagnostics === "all" ? "inherit" : "pipe",
    stderr: diagnostics === "all" ? "inherit" : "pipe",
  })
  const [code, stdout, stderr] = await Promise.all([
    child.exited,
    diagnostics === "all" ? Promise.resolve("") : new Response(child.stdout).text(),
    diagnostics === "all" ? Promise.resolve("") : new Response(child.stderr).text(),
  ])
  if (code !== 0) {
    const output = [stderr, stdout].filter((value) => value.trim().length > 0).join("\n").trim()
    throw new Error(`command failed with exit code ${code}: ${command.join(" ")}${output ? `\n${output}` : ""}`)
  }
}

export interface BuildLocalInferenceOptions {
  /** Print successful build diagnostics, or retain them only for a failed build. */
  readonly diagnostics?: "all" | "errors"
}

/**
 * Builds this machine's service with every backend of its host and stages
 * `inference/target/development/` in the release layout: `bin/`, `runtime/` (NVRTC on CUDA
 * hosts), `catalog/` and `installation.json`. The staged layout replaces the previous one atomically.
 */
export const buildLocalInference = async ({
  diagnostics = "all",
}: BuildLocalInferenceOptions = {}): Promise<{ readonly installationPath: string }> => {
  const host = hostById(currentHost())
  await run(["bun", "run", "catalog:build-bundle"], diagnostics)
  console.log(`[dev] Building the inference service for ${host.id}...`)
  const build = await buildInferenceBinary({ host, profile: "development", diagnostics })
  const target = resolve(INFERENCE_ROOT, "target")
  const staging = await mkdtemp(resolve(target, ".development-"))
  try {
    for (const directory of ["bin", "runtime", "catalog"]) {
      await mkdir(resolve(staging, directory), { recursive: true, mode: 0o700 })
    }
    await copyFile(build.binary, resolve(staging, inferenceExecutablePath(host)))
    for (const source of [...build.runtimeLibraries, ...build.runtimeNotices]) {
      await copyFile(source, resolve(staging, "runtime", basename(source)))
    }
    await copyFile(
      resolve(CATALOG_INPUTS, "model-planner-inputs.bundle"),
      resolve(staging, INFERENCE_PLANNER_BUNDLE),
    )
    await writeFile(
      resolve(staging, "installation.json"),
      `${Schema.encodeSync(Schema.parseJson(IcnInstallationDeclaration))({
        schemaVersion: 1,
        nativeBuild: build.identity.native_build,
      })}\n`,
    )
    await rm(DEVELOPMENT_INSTALLATION, { recursive: true, force: true })
    await rename(staging, DEVELOPMENT_INSTALLATION)
    return { installationPath: resolve(DEVELOPMENT_INSTALLATION, "installation.json") }
  } catch (cause) {
    await rm(staging, { recursive: true, force: true })
    throw cause
  }
}

if (import.meta.main) {
  const result = await buildLocalInference()
  if (process.argv.includes("--serve")) {
    const child = Bun.spawn([
      resolve(DEVELOPMENT_INSTALLATION, inferenceExecutablePath(hostById(currentHost()))),
      "serve",
      "--installation",
      result.installationPath,
      ...process.argv.slice(2).filter((argument) => argument !== "--serve"),
    ], {
      cwd: INFERENCE_ROOT,
      env: installationEnvironment(DEVELOPMENT_INSTALLATION),
      stdin: "inherit",
      stdout: "inherit",
      stderr: "inherit",
    })
    process.exit(await child.exited)
  }
  console.log(`Inference development installation ready at ${result.installationPath}`)
}
