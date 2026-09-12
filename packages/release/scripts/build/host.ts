import { acnExecutableRelativePath } from "../../src/macos-app"
import { buildMacApp } from "../apple/build-app"
import { buildDesktopApplication, DesktopBuildFailed } from "./desktop"
import { buildLinuxDesktopInstaller } from "./desktop-linux"
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
import { basename, delimiter, dirname, resolve } from "node:path"
import { Effect, Option, Schema } from "effect"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import {
  BackendEligibilityReport,
  IcnInstallationDeclaration,
  IcnStartupRecord,
} from "@magnitudedev/icn-protocol"
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
  releaseBuildEnvironment,
  type HostId,
} from "../../src/targets"
import { buildAcnBinary } from "./acn"
import { buildCliBinary } from "./cli"
import {
  buildArchive,
  type ArchiveSource,
  run,
  verifyAppleDeploymentTarget,
  verifyOwnedLoaderPaths,
} from "./common"
import { buildIcnBinary } from "../../../../inference/scripts/compile"
import { ACN_COORDINATION_REVISION } from "@magnitudedev/version"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/acn-protocol"
import { appleRequirement } from "../../src/trust"
import {
  ACN_EXECUTABLE_NAME,
  ICN_EXECUTABLE_NAME,
} from "../../src/executables"

const PROJECT_ROOT = resolve(import.meta.dir, "../../../..")

const smokeIcnServer = async (
  binary: string,
  installation: string,
  root: string,
  environment: Readonly<Record<string, string | undefined>>,
): Promise<void> => {
  const modelStore = resolve(root, "model-store")
  const cacheRoot = resolve(root, "cache")
  await Promise.all([
    mkdir(modelStore, { recursive: true, mode: 0o700 }),
    mkdir(cacheRoot, { recursive: true, mode: 0o700 }),
  ])
  const token = crypto.randomUUID()
  const instance = `release-smoke-${crypto.randomUUID()}`
  const child = Bun.spawn([
    binary,
    "serve",
    "--bind",
    "127.0.0.1:0",
    "--instance-id",
    instance,
    "--exit-on-stdin-eof",
    "--installation",
    installation,
    "--model-store",
    modelStore,
    "--cache-root",
    cacheRoot,
  ], {
    cwd: root,
    // Managed ICN lifetime guards require it to lead its own process group.
    detached: process.platform !== "win32",
    env: { ...environment, MAGNITUDE_ICN_AUTH_TOKEN: token },
    stdin: "pipe",
    stdout: "pipe",
    stderr: "inherit",
  })
  let reaped = false
  try {
    const reader = child.stdout.getReader()
    const decoder = new TextDecoder()
    const readiness = async (): Promise<{ readonly origin: string }> => {
      let pending = ""
      while (pending.length <= 64 * 1024) {
        const next = await reader.read()
        if (next.done) break
        pending += decoder.decode(next.value, { stream: true })
        let newline = pending.indexOf("\n")
        while (newline >= 0) {
          const line = pending.slice(0, newline).trimEnd()
          pending = pending.slice(newline + 1)
          const prefix = "MAGNITUDE_ICN_READY "
          if (line.startsWith(prefix)) {
            const value = Schema.decodeUnknownSync(
              Schema.parseJson(IcnStartupRecord),
            )(line.slice(prefix.length))
            if (value.instanceId !== instance || !value.origin) {
              throw new Error("ICN readiness record has the wrong identity")
            }
            return { origin: value.origin }
          }
          newline = pending.indexOf("\n")
        }
      }
      throw new Error("ICN exited without a bounded readiness record")
    }
    const ready = await Promise.race([
      readiness(),
      Bun.sleep(30_000).then(() => {
        throw new Error("ICN startup smoke timed out")
      }),
    ])
    const health = await fetch(`${ready.origin}/health`, {
      signal: AbortSignal.timeout(10_000),
    })
    if (!health.ok) throw new Error(`ICN health returned HTTP ${health.status}`)
    const healthBody = await health.json() as {
      readonly ready?: boolean
      readonly instanceId?: string
    }
    if (!healthBody.ready || healthBody.instanceId !== instance) {
      throw new Error("ICN health returned the wrong identity")
    }
    const hardware = await fetch(`${ready.origin}/api/v1/hardware`, {
      headers: { authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(10_000),
    })
    if (!hardware.ok) {
      throw new Error(`ICN authenticated hardware returned HTTP ${hardware.status}`)
    }
    const owned = process.platform === "win32" ? Option.none() : await Effect.runPromise(ProcessGroupControllerLive.inspect(child.pid))
    if (process.platform !== "win32" && Option.isNone(owned)) throw new Error("ICN disappeared before parent-loss acceptance")
    child.stdin.end()
    const exitCode = await Promise.race<number | undefined>([
      child.exited,
      Bun.sleep(5_000).then(() => undefined),
    ])
    if (exitCode === undefined) {
      throw new Error("ICN did not exit after its managed parent pipe closed")
    }
    reaped = true
    // EOF means owner loss, so the watchdog kills its group rather than exiting gracefully.
    if (exitCode !== 91 && child.signalCode !== "SIGKILL") {
      throw new Error(`ICN parent-loss watchdog exited with unexpected code ${exitCode} and signal ${child.signalCode}`)
    }
    if (Option.isSome(owned) && !await Effect.runPromise(ProcessGroupControllerLive.waitForGroupExit({ leader: owned.value }, "5 seconds"))) {
      throw new Error("ICN parent-loss watchdog left process-group members alive")
    }
  } finally {
    if (!reaped) {
      child.kill("SIGTERM")
      child.stdin.end()
      const exited = await Promise.race([
        child.exited.then(() => true),
        Bun.sleep(5_000).then(() => false),
      ])
      if (!exited) {
        child.kill("SIGKILL")
        await child.exited
      }
    }
  }
}

export const smokeHostArchives = async (
  host: ReturnType<typeof hostById>,
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
      backend: "cpu",
      nativeBuild: Option.getOrThrow(icnArtifact.nativeBuild),
      backendModuleAbi: Option.getOrThrow(icnArtifact.backendModuleAbi),
    })}\n`)
    const environment = host.id.startsWith("windows-")
      ? {
        ...process.env,
        PATH: [resolve(icnRoot, "runtime"), process.env.PATH]
          .filter(Boolean)
          .join(delimiter),
      }
      : {
        ...process.env,
        ...(host.id.startsWith("darwin-")
          ? { DYLD_LIBRARY_PATH: "" }
          : { LD_LIBRARY_PATH: "" }),
      }
    const icnBinary = resolve(icnRoot, `bin/${ICN_EXECUTABLE_NAME}${extension}`)
    await smokeIcnServer(icnBinary, declaration, icnRoot, environment)
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
    await rm(root, { recursive: true, force: true })
  }
}

const regularSources = (
  directory: "runtime" | "backends",
  files: readonly string[],
): readonly ArchiveSource[] => {
  const seen = new Set<string>()
  return files.map((source) => {
    const name = basename(source)
    if (seen.has(name)) {
      throw new Error(`duplicate ${directory} output ${name}`)
    }
    seen.add(name)
    return { path: `${directory}/${name}`, source, mode: 0o755 }
  })
}

export const buildHostArtifacts = async (
  hostId: HostId,
  catalogRoot: string,
  outputRoot: string,
): Promise<void> => {
  const host = hostById(hostId)
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
  const icn = await buildIcnBinary({
    target: host.bunTarget,
    profile: `base-${host.id}`,
    features: host.cargoFeatures,
    buildEnvironment: releaseBuildEnvironment(host),
  })
  const cpuModules = icn.backendModules.filter((file) =>
    basename(file).toLowerCase().includes("cpu")
  )
  if (cpuModules.length === 0) {
    throw new Error(`${host.id} ICN base emitted no CPU module`)
  }
  await verifyOwnedLoaderPaths({
    host: host.id,
    executable: icn.binary,
    modules: cpuModules,
    runtime: icn.runtimeLibraries,
  })
  await verifyAppleDeploymentTarget(host.id, [
    cli,
    acn,
    icn.binary,
    ...icn.runtimeLibraries,
    ...cpuModules,
  ])

  if (host.id.startsWith("darwin-")) {
    for (const kind of ["cli", "acn"]) {
      const embedded = await runAppleBuild(regularAppleFiles(resolve(PROJECT_ROOT, "bin/apple-inputs", kind)))
      await verifyAppleDeploymentTarget(host.id, embedded.map((file) => file.source))
    }
    for (const file of [icn.binary, ...icn.runtimeLibraries, ...cpuModules]) {
      await runAppleBuild(signAppleCode(file, `dev.magnitude.inference.${basename(file)}`, file === icn.binary ? "native" : "library"))
    }
  }
  await chmod(cli, 0o755)
  await chmod(acn, 0o755)
  await chmod(icn.binary, 0o755)

  const loader = host.id.startsWith("windows-")
    ? "PATH"
    : host.id.startsWith("darwin-")
      ? "DYLD_LIBRARY_PATH"
      : "LD_LIBRARY_PATH"
  const eligibility = await run([
    icn.binary,
    "backend-eligibility",
    "--json",
  ], {
    env: {
      ...process.env,
      [loader]: [...icn.runtimeLibraries.map(dirname), process.env[loader]]
        .filter(Boolean)
        .join(delimiter),
    },
  })
  Schema.decodeUnknownSync(
    Schema.parseJson(BackendEligibilityReport),
  )(eligibility)

  const cliArchivePath = resolve(output, cliArchive(host.id))
  const acnArchivePath = resolve(output, acnArchive(host.id))
  const icnArchivePath = resolve(output, icnBaseArchive(host.id))
  const cliNotary = host.id.startsWith("darwin-")
    ? await runAppleBuild(notarizeAppleUnit("cli", output, [cli, resolve(PROJECT_ROOT, "bin/apple-inputs/cli")])) : Option.none()
  const icnNotary = host.id.startsWith("darwin-")
    ? await runAppleBuild(notarizeAppleUnit("inference", output, [icn.binary, ...icn.runtimeLibraries, ...cpuModules])) : Option.none()
  const cliArtifact = await buildArchive(
    cliArchivePath,
    resolve(output, `cli-${host.id}.artifact.json`),
    {
      id: `cli-${host.id}`,
      kind: "cli",
      host: Option.some(host.id),
      backend: Option.none(),
      requiredBaseId: Option.none(),
      nativeBuild: Option.none(),
      backendModuleAbi: Option.none(),
      compatibility: Option.none(),
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
    await run(["bun", "run", "build"], { cwd: resolve(PROJECT_ROOT, "desktop") })
    const packages = await runAppleBuild(buildDesktopApplication({ service: acn, outputDirectory: resolve(output, ".desktop-build"), version, revision: ACN_COORDINATION_REVISION }))
    if (packages.length !== 1) throw new Error("Desktop packaging did not produce exactly one host application")
    const desktop = await runAppleBuild(buildDesktopDmg({ app: resolve(packages[0]!, "Magnitude.app"), output, host: Schema.decodeUnknownSync(Schema.Literal("darwin-arm64", "darwin-x64"))(host.id) }))
    desktopArtifacts.push(desktop.artifact, desktop.updateArtifact)
    notarizations.push(desktop.notarization)
  } else if (host.id.startsWith("linux-")) {
    await run(["bun", "run", "build"], { cwd: resolve(PROJECT_ROOT, "desktop") })
    const installers = await Effect.runPromise(Effect.gen(function* () {
      const arch = host.id === "linux-arm64-gnu" ? "arm64" : "x64"
      const applications = yield* buildDesktopApplication({
        service: acn, outputDirectory: resolve(output, ".desktop-build"),
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
  }
  const acnArtifact = await buildArchive(
    acnArchivePath,
    resolve(output, `acn-${host.id}.artifact.json`),
    {
      id: `acn-${host.id}`,
      kind: "acn",
      host: Option.some(host.id),
      backend: Option.none(),
      requiredBaseId: Option.none(),
      nativeBuild: Option.none(),
      backendModuleAbi: Option.none(),
      compatibility: Option.none(),
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
      backend: Option.some("cpu"),
      requiredBaseId: Option.none(),
      nativeBuild: Option.some(icn.identity.native_build),
      backendModuleAbi: Option.some(icn.identity.backend_module_abi),
      compatibility: Option.none(),
    },
    [
      {
        path: `bin/${ICN_EXECUTABLE_NAME}${host.executableExtension}`,
        source: icn.binary,
        mode: 0o755,
      },
      {
        path: "catalog/model-planner-inputs.bundle",
        source: resolve(catalogRoot, "model-planner-inputs.bundle"),
        mode: 0o644,
      },
      ...regularSources("runtime", icn.runtimeLibraries),
      ...regularSources("backends", cpuModules),
    ],
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
