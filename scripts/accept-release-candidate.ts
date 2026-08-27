import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import {
  AcnBoundary,
  AcnRpc,
  BunDetachedChildProcessSpawner,
  ChildProcessSpawner,
  AcnInstanceManager,
  makeAcnJitRuntime,
  makeLocalAcnInstanceManager,
} from "@magnitudedev/sdk"
import { BunSqliteDriverLayer } from "@magnitudedev/sdk/bun"
import { Duration, Effect, Exit, Layer, Schema, Scope } from "effect"
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve } from "node:path"
import { releaseUrl } from "@magnitudedev/release/acquisition"
import { ReleaseManifestSchema } from "@magnitudedev/release/contracts"

const BOOTSTRAP_TIMEOUT_MS = 2 * 60_000
const SHUTDOWN_TIMEOUT_MS = 20_000

const candidate = resolve(process.argv[2] ?? "release-candidate")
const tarballArgument = process.argv[3]
if (!tarballArgument) {
  throw new Error("accepted npm tarball is required")
}
const tarball = resolve(tarballArgument)

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
    throw new Error(
      `${command[0]} failed with exit ${code}: ${(stderr || stdout).trim()}`,
    )
  }
  return stdout
}

const manifest = Schema.decodeUnknownSync(
  Schema.parseJson(ReleaseManifestSchema),
)(await readFile(resolve(candidate, "magnitude-release.json"), "utf8"))

const routes = new Map(
  [
    "magnitude-release.json",
    ...manifest.artifacts.map((artifact) => artifact.filename),
  ].map((name) => [
    new URL(releaseUrl("http://release.invalid", manifest.version, name)).pathname,
    name,
  ]),
)
const server = Bun.serve({
  port: 0,
  async fetch(request) {
    const name = routes.get(new URL(request.url).pathname)
    if (!name) return new Response("missing", { status: 404 })
    try {
      return new Response(await readFile(resolve(candidate, name)))
    } catch {
      return new Response("missing", { status: 404 })
    }
  },
})
const baseUrl = `http://127.0.0.1:${server.port}`
const root = await mkdtemp(resolve(tmpdir(), "magnitude-candidate-"))
const dataDir = resolve(root, "home-bootstrap", ".magnitude")
const ensurerScope = await Effect.runPromise(Scope.make())

const manager = await Effect.runPromise(
  makeLocalAcnInstanceManager({ dataDir }).pipe(
    Effect.provideService(ChildProcessSpawner, BunDetachedChildProcessSpawner),
    Effect.provideService(Scope.Scope, ensurerScope),
    Effect.provide([
      BunContext.layer,
      FetchHttpClient.layer,
      BunSqliteDriverLayer,
    ]),
  ),
)

const environment = (home: string) => ({
  ...process.env,
  HOME: home,
  USERPROFILE: home,
  MAGNITUDE_ACN_VERSION: manifest.version,
  MAGNITUDE_RELEASE_BASE_URL: baseUrl,
})

const processIsAlive = (pid: number): boolean => {
  try {
    process.kill(pid, 0)
    return true
  } catch (cause) {
    return cause instanceof Error &&
      "code" in cause &&
      cause.code === "EPERM"
  }
}

const processTree = async (): Promise<ReadonlyMap<number, readonly number[]>> => {
  const output = await run(["ps", "-axo", "pid=,ppid="])
  const children = new Map<number, number[]>()
  for (const line of output.split("\n")) {
    const [pidText, parentText] = line.trim().split(/\s+/)
    const pid = Number(pidText)
    const parent = Number(parentText)
    if (!Number.isSafeInteger(pid) || !Number.isSafeInteger(parent)) continue
    const existing = children.get(parent)
    if (existing) existing.push(pid)
    else children.set(parent, [pid])
  }
  return children
}

const descendantsOf = (
  rootPid: number,
  tree: ReadonlyMap<number, readonly number[]>,
): readonly number[] => {
  const descendants: number[] = []
  const pending = [...(tree.get(rootPid) ?? [])]
  while (pending.length > 0) {
    const pid = pending.pop()
    if (pid === undefined) continue
    descendants.push(pid)
    pending.push(...(tree.get(pid) ?? []))
  }
  return descendants
}

const registeredProcess = async (pid: number): Promise<{
  readonly pid: number
  readonly descendants: readonly number[]
}> => {
  if (!processIsAlive(pid)) {
    throw new Error(`release bootstrap ACN ${pid} exited before teardown`)
  }
  return {
    pid,
    descendants: descendantsOf(pid, await processTree()),
  }
}

const terminateBootstrap = async (pid?: number): Promise<void> => {
  const registered = pid === undefined ? undefined : await registeredProcess(pid)
  let terminationFailure: unknown
  try {
    await Effect.runPromise(manager.stop)
  } catch (cause) {
    terminationFailure = cause
  }
  if (registered === undefined) {
    if (terminationFailure !== undefined) throw terminationFailure
    return
  }
  const ownedProcesses = [registered.pid, ...registered.descendants]
  const deadline = Date.now() + SHUTDOWN_TIMEOUT_MS
  while (
    ownedProcesses.some(processIsAlive) &&
    Date.now() < deadline
  ) {
    await Bun.sleep(100)
  }
  const survivors = ownedProcesses.filter(processIsAlive)
  if (survivors.length === 0) {
    if (terminationFailure !== undefined) throw terminationFailure
    return
  }
  for (const pid of survivors) {
    try {
      process.kill(pid, "SIGKILL")
    } catch {
      // The process exited between observation and cleanup.
    }
  }
  throw new Error(
    `release bootstrap left processes alive after shutdown: ${survivors.join(", ")}`,
  )
}

const probeBootstrap = Effect.gen(function* () {
  const runtime = yield* makeAcnJitRuntime().pipe(
    Effect.provideService(AcnInstanceManager, manager),
  )

  yield* runtime.startup.prepare
  const protocolLayer = runtime.protocolLayer.pipe(
    Layer.provide(FetchHttpClient.layer),
  )
  return yield* Effect.gen(function* () {
    const client = yield* AcnRpc.makeRpcClient(AcnBoundary)
    return yield* client.Health({})
  }).pipe(
    Effect.provide(protocolLayer),
    Effect.scoped,
  )
}).pipe(
  Effect.provide(FetchHttpClient.layer),
  Effect.scoped,
  Effect.timeout(Duration.millis(BOOTSTRAP_TIMEOUT_MS)),
)

const acceptBootstrap = async (): Promise<void> => {
  let accepted = false
  let healthPid: number | undefined
  try {
    const health = await Effect.runPromise(probeBootstrap)
    if (
      health.service !== "magnitude-acn" ||
      health.version !== manifest.version ||
      health.revision !== manifest.acnRevision ||
      health.state._tag !== "Ready"
    ) {
      throw new Error(
        `release bootstrap returned incompatible health: ${JSON.stringify(health)}`,
      )
    }
    healthPid = health.pid
    await registeredProcess(health.pid)
    accepted = true
  } finally {
    try {
      await terminateBootstrap(healthPid)
    } catch (cause) {
      if (accepted) throw cause
    }
  }
}

const invoke = async (
  command: readonly string[],
  directory: string,
  home: string,
): Promise<void> => {
  const output = (await run(command, {
    cwd: directory,
    env: environment(home),
  })).trim()
  if (output !== manifest.version) {
    throw new Error(`${command[0]} returned ${output}; expected ${manifest.version}`)
  }
}

process.env.MAGNITUDE_ACN_VERSION = manifest.version
process.env.MAGNITUDE_RELEASE_BASE_URL = baseUrl

try {
  const npmRoot = resolve(root, "npm")
  const bunRoot = resolve(root, "bun")
  await mkdir(npmRoot)
  await mkdir(bunRoot)
  await writeFile(resolve(npmRoot, "package.json"), "{}\n")
  await writeFile(resolve(bunRoot, "package.json"), "{}\n")
  await run(["npm", "install", "--ignore-scripts", tarball], { cwd: npmRoot })
  await invoke(
    ["npx", "--no-install", "magnitude", "--version"],
    npmRoot,
    resolve(root, "home-npx"),
  )
  await run(["bun", "add", "--ignore-scripts", tarball], { cwd: bunRoot })
  await invoke(
    ["bunx", "--bun", "magnitude", "--version"],
    bunRoot,
    resolve(root, "home-bunx"),
  )

  await acceptBootstrap()
} finally {
  await Effect.runPromise(Scope.close(ensurerScope, Exit.void))
  server.stop(true)
  await rm(root, { recursive: true, force: true })
}
