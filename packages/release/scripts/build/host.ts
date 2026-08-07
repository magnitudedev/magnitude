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
import { Option, Schema } from "effect"
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
  cliArchive,
  currentHost,
  hostById,
  icnBaseArchive,
  type HostId,
} from "../../src/targets"
import { buildAcnBinary } from "./acn"
import { buildCliBinary } from "./cli"
import {
  buildArchive,
  type ArchiveSource,
  run,
  verifyOwnedLoaderPaths,
} from "./common"
import { buildIcnBinary } from "../../../../inference/scripts/compile"
import { ACN_COORDINATION_REVISION } from "@magnitudedev/version"

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
    const hardware = await fetch(`${ready.origin}/v1/hardware`, {
      headers: { authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(10_000),
    })
    if (!hardware.ok) {
      throw new Error(`ICN authenticated hardware returned HTTP ${hardware.status}`)
    }
    child.stdin.end()
    const exitCode = await Promise.race<number | undefined>([
      child.exited,
      Bun.sleep(5_000).then(() => undefined),
    ])
    if (exitCode === undefined) {
      throw new Error("ICN did not exit after its managed parent pipe closed")
    }
    reaped = true
    if (exitCode !== 0) {
      throw new Error(`ICN parent-pipe shutdown exited with code ${exitCode}`)
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

const smokeHostArchives = async (
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
      await readFile(resolve(PROJECT_ROOT, "packages/cli/package.json"), "utf8"),
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
        resolve(acnRoot, `bin/magnitude-acn${extension}`),
        "version",
      ])).trim() !== version
    ) throw new Error(`${host.id} ACN archive returned the wrong version`)
    if (
      Number((await run([
        resolve(acnRoot, `bin/magnitude-acn${extension}`),
        "coordination-revision",
      ])).trim()) !== ACN_COORDINATION_REVISION
    ) throw new Error(`${host.id} ACN archive returned the wrong coordination revision`)
    if (!(await run([
      resolve(acnRoot, `bin/magnitude-acn${extension}`),
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
    const icnBinary = resolve(icnRoot, `bin/magnitude-icn${extension}`)
    await smokeIcnServer(icnBinary, declaration, icnRoot, environment)
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

  if (host.id.startsWith("darwin-")) {
    for (const file of [icn.binary, ...icn.runtimeLibraries, ...cpuModules]) {
      await run(["codesign", "--force", "--sign", "-", file])
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
  await buildArchive(
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
  await buildArchive(
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
    [{
      path: `bin/magnitude-acn${host.executableExtension}`,
      source: acn,
      mode: 0o755,
    }],
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
        path: `bin/magnitude-icn${host.executableExtension}`,
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
}

if (import.meta.main) {
  const hostId = (process.argv[2] as HostId | undefined) ?? currentHost()
  await buildHostArtifacts(
    hostId,
    resolve(process.argv[3] ?? "inference/target/catalog-inputs"),
    resolve(process.argv[4] ?? `release/${hostId}`),
  )
}
