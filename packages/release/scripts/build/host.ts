import { acnExecutableRelativePath } from "../../src/macos-app"
import { buildMacApp } from "../apple/build-app"
import { buildDesktopApplication, DesktopBuildFailed } from "./desktop"
import { buildLinuxDesktopInstaller } from "./desktop-linux"
import { buildWindowsDesktopInstaller } from "./desktop-windows"
import { signWindowsCode } from "./windows-signing"
import { BunContext } from "@effect/platform-bun"
import { buildDesktopDmg, validateDesktopDistribution } from "../apple/desktop"
import { appleSigning, signAppleCode, appleCommand } from "../apple/signing"
import { runAppleBuild } from "../apple/compile-bun"
import { notarizeAppleUnit, regularAppleFiles, writeAppleReceipt } from "../apple/distribution"
import {
  chmod,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import { basename, dirname, resolve } from "node:path"
import { Effect, Option, Schedule, Schema } from "effect"
import { IcnInstallationDeclaration } from "@magnitudedev/icn-protocol"
import {
  type ReleaseArtifact,
} from "../../src/contracts"
import {
  acnArchive,
  desktopInstaller,
  desktopUpdateArchive,
  cliArchive,
  currentHost,
  hostById,
  icnBaseArchive,
  type HostId,
  type ReleaseHost,
} from "../../src/targets"
import { buildAcnBinary } from "./acn"
import { desktopBuildEnvironment } from "./desktop-distribution"
import { buildCliBinary } from "./cli"
import {
  buildArchive,
  type ArchiveSource,
  run,
  verifyAppleDeploymentTarget,
  verifyOwnedLoaderPaths,
} from "./common"
import { buildInferenceBinary, type InferenceBuild } from "../../../../inference/scripts/compile"
import { smokeInstallation } from "../../../../inference/scripts/smoke"
import { ACN_COORDINATION_REVISION } from "@magnitudedev/version"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/acn-protocol"
import { appleRequirement } from "../../src/trust"
import { ACN_EXECUTABLE_NAME } from "../../src/executables"
import {
  INFERENCE_PLANNER_BUNDLE,
  inferenceExecutablePath,
  inferenceRequiredPaths,
} from "../../src/inference-installation"

const PROJECT_ROOT = resolve(import.meta.dir, "../../../..")

export const smokeHostArchives = async (
  host: ReleaseHost,
  cliArchivePath: string,
  acnArchivePath: string,
  icnArchivePath: string,
  icnArtifact: ReleaseArtifact,
): Promise<void> => {
  const root = await mkdtemp(resolve(tmpdir(), `magnitude-${host.id}-`))
  try {
    const [cliRoot, acnRoot, icnRoot] = ["cli", "acn", "icn"].map((name) =>
      resolve(root, name)
    )
    await Promise.all([cliRoot, acnRoot, icnRoot].map((directory) =>
      mkdir(directory, { recursive: true, mode: 0o700 })
    ))
    await run(["tar", "-xzf", cliArchivePath, "-C", cliRoot])
    await run(["tar", "-xzf", acnArchivePath, "-C", acnRoot])
    await run(["tar", "-xzf", icnArchivePath, "-C", icnRoot])

    const packageJson = JSON.parse(
      await readFile(resolve(PROJECT_ROOT, "packages/launcher/package.json"), "utf8"),
    ) as { readonly version?: string }
    const version = packageJson.version
    if (!version) throw new Error("CLI package has no version")
    const extension = host.executableExtension
    if (
      (await run([
        resolve(cliRoot, `bin/magnitude-cli${extension}`),
        "--version",
      ])).trim() !== version
    ) throw new Error(`${host.id} CLI archive returned the wrong version`)
    if (
      (await run([
        resolve(acnRoot, acnExecutableRelativePath(host.id)),
        "version",
      ])).trim() !== version
    ) throw new Error(`${host.id} ACN archive returned the wrong version`)
    if (
      Number((await run([
        resolve(acnRoot, acnExecutableRelativePath(host.id)),
        "coordination-revision",
      ])).trim()) !== ACN_COORDINATION_REVISION
    ) throw new Error(`${host.id} ACN archive returned the wrong coordination revision`)
    if (!(await run([
      resolve(acnRoot, acnExecutableRelativePath(host.id)),
      "doctor",
    ])).includes("ripgrep")) {
      throw new Error(`${host.id} ACN archive has no working embedded ripgrep`)
    }

    const declaration = resolve(icnRoot, "installation.json")
    await writeFile(declaration, `${Schema.encodeSync(
      Schema.parseJson(IcnInstallationDeclaration),
    )({
      schemaVersion: 1,
      nativeBuild: Option.getOrThrow(icnArtifact.nativeBuild),
    })}\n`)
    await smokeInstallation(declaration)
    if (host.id.startsWith("darwin-")) {
      await runAppleBuild(validateDesktopDistribution({
        image: resolve(dirname(acnArchivePath), desktopInstaller(Schema.decodeUnknownSync(Schema.Literal("darwin-arm64", "darwin-x64"))(host.id))),
        updateArchive: resolve(dirname(acnArchivePath), desktopUpdateArchive(Schema.decodeUnknownSync(Schema.Literal("darwin-arm64", "darwin-x64"))(host.id))),
        version, revision: ACN_COORDINATION_REVISION, rpcVersion: MAGNITUDE_RPC_VERSION, inferenceInstallation: declaration,
      }))
      await run([resolve(cliRoot, "bin/magnitude-cli"), "native-runtime-check"])
      const app = resolve(acnRoot, "Magnitude.app")
      const signing = await runAppleBuild(appleSigning)
      await run(["/usr/bin/codesign", "--verify", "--deep", "--strict", "-R", `=${appleRequirement("dev.magnitude.service", signing.team)}`, app])
      await run(["/usr/bin/codesign", "--verify", "--strict", "-R", `=${appleRequirement("dev.magnitude.cli", signing.team)}`, resolve(cliRoot, "bin/magnitude-cli")])
      if (signing.mode === "developer-id") await run(["/usr/bin/xcrun", "stapler", "validate", app])
    }
  } finally {
    await Effect.runPromise(Effect.tryPromise(() => rm(root, { recursive: true, force: true })).pipe(
      Effect.retry({ times: 6, schedule: Schedule.spaced("500 millis"), while: error => process.platform === "win32" &&
        Schema.is(Schema.Struct({ code: Schema.Literal("EPERM", "EBUSY", "ENOTEMPTY") }))(error.cause) }),
    ))
  }
}

/** The one inference artifact of a host: the service, its runtime directory and the planner inputs. */
export const inferenceArchiveSources = (
  host: ReleaseHost,
  inference: InferenceBuild,
  plannerBundle: string,
): readonly ArchiveSource[] => {
  const sources: readonly ArchiveSource[] = [
    { path: inferenceExecutablePath(host), source: inference.binary, mode: 0o755 },
    { path: INFERENCE_PLANNER_BUNDLE, source: plannerBundle, mode: 0o644 },
    ...inference.runtimeLibraries.map((source) => ({ path: `runtime/${basename(source)}`, source, mode: 0o755 })),
    ...inference.runtimeNotices.map((source) => ({ path: `runtime/${basename(source)}`, source, mode: 0o644 })),
  ]
  const paths = new Set(sources.map((source) => source.path))
  if (paths.size !== sources.length) throw new Error(`${host.id} inference artifact has duplicate files`)
  const missing = inferenceRequiredPaths(host).filter((path) => !paths.has(path))
  if (missing.length > 0) throw new Error(`${host.id} inference artifact is missing ${missing.join(", ")}`)
  return sources
}

export const buildHostArtifacts = async (
  hostId: HostId,
  catalogRoot: string,
  outputRoot: string,
): Promise<void> => {
  const host = hostById(hostId)
  // Resolved before compiling so a missing or malformed publisher identity fails immediately.
  const desktopEnvironment = await Effect.runPromise(desktopBuildEnvironment(host.id))
  const output = resolve(outputRoot)
  await rm(output, { recursive: true, force: true })
  await mkdir(output, { recursive: true, mode: 0o700 })

  await run([
    "bun",
    "run",
    resolve(PROJECT_ROOT, "packages/version/scripts/generate-version.ts"),
  ], { cwd: PROJECT_ROOT })
  const cli = await buildCliBinary(host.bunTarget)
  const acn = await buildAcnBinary(host.bunTarget)
  const inference = await buildInferenceBinary({ host, profile: "release", diagnostics: "all" })
  await verifyOwnedLoaderPaths({
    host: host.id,
    executable: inference.binary,
    runtime: inference.runtimeLibraries,
  })
  await verifyAppleDeploymentTarget(host.id, [cli, acn, inference.binary])

  if (host.id.startsWith("darwin-")) {
    for (const kind of ["cli", "acn"]) {
      const embedded = await runAppleBuild(regularAppleFiles(resolve(PROJECT_ROOT, "bin/apple-inputs", kind)))
      await verifyAppleDeploymentTarget(host.id, embedded.map((file) => file.source))
    }
    await runAppleBuild(signAppleCode(inference.binary, `dev.magnitude.inference.${basename(inference.binary)}`, "native"))
  }
  await chmod(cli, 0o755)
  await chmod(acn, 0o755)
  await chmod(inference.binary, 0o755)

  const cliArchivePath = resolve(output, cliArchive(host.id))
  if (host.id === "windows-x64-msvc") {
    // NVIDIA's NVRTC and Microsoft's CRT are redistributed unmodified.
    await Effect.runPromise(Effect.forEach([cli, acn, inference.binary], signWindowsCode, { discard: true }).pipe(Effect.provide(BunContext.layer)))
  }
  const acnArchivePath = resolve(output, acnArchive(host.id))
  const icnArchivePath = resolve(output, icnBaseArchive(host.id))
  const cliNotary = host.id.startsWith("darwin-")
    ? await runAppleBuild(notarizeAppleUnit("cli", output, [cli, resolve(PROJECT_ROOT, "bin/apple-inputs/cli")])) : Option.none()
  const icnNotary = host.id.startsWith("darwin-")
    ? await runAppleBuild(notarizeAppleUnit("inference", output, [inference.binary])) : Option.none()
  const cliArtifact = await buildArchive(
    cliArchivePath,
    resolve(output, `cli-${host.id}.artifact.json`),
    {
      id: `cli-${host.id}`,
      kind: "cli",
      host: Option.some(host.id),
      nativeBuild: Option.none(),
    },
    [{
      path: `bin/magnitude-cli${host.executableExtension}`,
      source: cli,
      mode: 0o755,
    }],
  )
  let acnSources: readonly ArchiveSource[] = [{ path: `bin/${ACN_EXECUTABLE_NAME}${host.executableExtension}`, source: acn, mode: 0o755 }]
  const notarizations = [cliNotary, icnNotary]
  const desktopArtifacts: ReleaseArtifact[] = []
  const version = Schema.decodeUnknownSync(Schema.parseJson(Schema.Struct({ version: Schema.NonEmptyString })))(
    await readFile(resolve(PROJECT_ROOT, "packages/launcher/package.json"), "utf8"),
  ).version
  if (host.id.startsWith("darwin-")) {
    const appRoot = resolve(output, ".app-build")
    const app = await runAppleBuild(buildMacApp(appRoot, acn, version, ACN_COORDINATION_REVISION))
    notarizations.push(await runAppleBuild(notarizeAppleUnit("app", output, [app, resolve(PROJECT_ROOT, "bin/apple-inputs/acn")])))
    if ((await runAppleBuild(appleSigning)).mode === "developer-id") {
      await runAppleBuild(appleCommand("/usr/bin/xcrun", "stapler", "staple", app))
      await runAppleBuild(appleCommand("/usr/bin/xcrun", "stapler", "validate", app))
    }
    acnSources = (await runAppleBuild(regularAppleFiles(app))).map((file) => ({ ...file, path: `Magnitude.app/${file.path}` }))
    await run(["bun", "run", "build"], { cwd: resolve(PROJECT_ROOT, "desktop"), env: desktopEnvironment })
    const packages = await runAppleBuild(buildDesktopApplication({ service: acn, cli, outputDirectory: resolve(output, ".desktop-build"), version, revision: ACN_COORDINATION_REVISION }))
    if (packages.length !== 1) throw new Error("Desktop packaging did not produce exactly one host application")
    const desktop = await runAppleBuild(buildDesktopDmg({ app: resolve(packages[0]!, "Magnitude.app"), output, host: Schema.decodeUnknownSync(Schema.Literal("darwin-arm64", "darwin-x64"))(host.id) }))
    desktopArtifacts.push(desktop.artifact, desktop.updateArtifact)
    notarizations.push(desktop.notarization)
  } else if (host.id.startsWith("linux-")) {
    await run(["bun", "run", "build"], { cwd: resolve(PROJECT_ROOT, "desktop"), env: desktopEnvironment })
    const installers = await Effect.runPromise(Effect.gen(function* () {
      const arch = host.id === "linux-arm64-gnu" ? "arm64" : "x64"
      const applications = yield* buildDesktopApplication({
        service: acn, cli, outputDirectory: resolve(output, ".desktop-build"),
        version, revision: ACN_COORDINATION_REVISION, target: { platform: "linux", arch },
      })
      if (applications.length !== 1) return yield* new DesktopBuildFailed({ message: "Desktop packaging did not produce exactly one Linux application" })
      const artifacts: ReleaseArtifact[] = []
      for (const format of ["deb", "rpm"] as const) {
        const installer = yield* buildLinuxDesktopInstaller({
          app: applications[0]!, output, arch, format, version, revision: ACN_COORDINATION_REVISION,
        })
        artifacts.push(installer.artifact)
      }
      return artifacts
    }).pipe(Effect.provide(BunContext.layer)))
    desktopArtifacts.push(...installers)
  } else if (host.id === "windows-x64-msvc") {
    await run(["bun", "run", "build"], { cwd: resolve(PROJECT_ROOT, "desktop"), env: desktopEnvironment })
    const guard = resolve(output, ".desktop-build", "MagnitudeInstallGuard.dll")
    await run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
      resolve(PROJECT_ROOT, "packages/release/scripts/build/windows-installer.ps1"), "-Output", guard])
    const installer = await Effect.runPromise(Effect.gen(function* () {
      const applications = yield* buildDesktopApplication({
        service: acn, cli, outputDirectory: resolve(output, ".desktop-build", "application"),
        version, revision: ACN_COORDINATION_REVISION, target: { platform: "win32", arch: "x64" },
      })
      if (applications.length !== 1) return yield* new DesktopBuildFailed({ message: "Desktop packaging did not produce exactly one Windows application" })
      return yield* buildWindowsDesktopInstaller({
        app: applications[0]!, guard, makensis: "makensis.exe", version,
        revision: ACN_COORDINATION_REVISION, output,
      })
    }).pipe(Effect.provide(BunContext.layer)))
    desktopArtifacts.push(installer.artifact)
  }
  const acnArtifact = await buildArchive(
    acnArchivePath,
    resolve(output, `acn-${host.id}.artifact.json`),
    {
      id: `acn-${host.id}`,
      kind: "acn",
      host: Option.some(host.id),
      nativeBuild: Option.none(),
    },
    acnSources,
  )
  const icnArtifact = await buildArchive(
    icnArchivePath,
    resolve(output, `icn-base-${host.id}.artifact.json`),
    {
      id: `icn-base-${host.id}`,
      kind: "icn-base",
      host: Option.some(host.id),
      nativeBuild: Option.some(inference.identity.native_build),
    },
    inferenceArchiveSources(host, inference, resolve(catalogRoot, "model-planner-inputs.bundle")),
  )
  await smokeHostArchives(
    host,
    cliArchivePath,
    acnArchivePath,
    icnArchivePath,
    icnArtifact,
  )
  if (host.id.startsWith("darwin-")) {
    await runAppleBuild(writeAppleReceipt(output, [cliArtifact, acnArtifact, icnArtifact, ...desktopArtifacts], notarizations, true))
    await rm(resolve(output, ".app-build"), { recursive: true, force: true })
  }
  await rm(resolve(output, ".desktop-build"), { recursive: true, force: true })
}

if (import.meta.main) {
  const hostId = (process.argv[2] as HostId | undefined) ?? currentHost()
  await buildHostArtifacts(
    hostId,
    resolve(process.argv[3] ?? "inference/target/catalog-inputs"),
    resolve(process.argv[4] ?? `release/${hostId}`),
  )
}
