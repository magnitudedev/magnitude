import { execFile } from "node:child_process"
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { promisify } from "node:util"
import { Database } from "bun:sqlite"
import {
  AcnInstanceIdSchema,
  ModelCatalogStateSchema,
  ProcessStartIdentitySchema,
  SDK_REVISION,
  SDK_VERSION,
  type ModelCatalogState,
  type ModelResidency,
} from "@magnitudedev/sdk"
import { Option, Schema } from "effect"
import { afterAll, beforeAll, describe, expect, it } from "vitest"
import { makeModel, TEST_MODEL_ID } from "../features/local-inference/test-fixtures"

const execFileAsync = promisify(execFile)
const cliRunner = resolve(import.meta.dirname, "fixtures/run-cli.ts")
const CLI_TIMEOUT_MS = 10_000
let transportTrace: string[] = []

interface CliResult {
  readonly exitCode: number
  readonly stdout: string
  readonly stderr: string
}

interface RpcRequest {
  readonly id: string
  readonly tag: string
  readonly payload?: unknown
}

interface RecordedRpc {
  readonly tag: string
  readonly payload?: unknown
}

const processStartIdentity = async (pid: number): Promise<string> => {
  if (process.platform === "linux") {
    const stat = await readFile(`/proc/${pid}/stat`, "utf8")
    const close = stat.lastIndexOf(")")
    if (close < 0) throw new Error("Malformed Linux process stat")
    const startTicks = stat.slice(close + 2).trim().split(/\s+/)[19]
    if (startTicks === undefined) throw new Error("Linux process stat has no start time")
    const bootId = (await readFile("/proc/sys/kernel/random/boot_id", "utf8")).trim().toLowerCase()
    return `linux:${bootId}:${startTicks}`
  }
  if (process.platform === "darwin") {
    const [{ stdout: started }, { stdout: boot }] = await Promise.all([
      execFileAsync("/bin/ps", ["-o", "lstart=", "-p", String(pid)]),
      execFileAsync("/usr/sbin/sysctl", ["-n", "kern.bootsessionuuid"]),
    ])
    return `darwin:${boot.trim().toLowerCase()}:${started.trim()}`
  }
  if (process.platform === "win32") {
    const { stdout } = await execFileAsync("powershell.exe", [
      "-NoProfile",
      "-NonInteractive",
      "-Command",
      `$p = Get-Process -Id ${pid} -ErrorAction Stop; $p.StartTime.ToUniversalTime().Ticks`,
    ])
    return `windows:${stdout.trim()}`
  }
  throw new Error(`Unsupported process platform ${process.platform}`)
}

const writeOwnerRecord = async (home: string, port: number): Promise<void> => {
  const directory = join(home, ".magnitude", "acn")
  await mkdir(directory, { recursive: true, mode: 0o700 })
  const database = new Database(join(directory, "coordination.sqlite"), { create: true })
  try {
    database.exec(`CREATE TABLE owner (
      id INTEGER PRIMARY KEY CHECK (id = 1),
      pid INTEGER NOT NULL CHECK (pid > 0 AND pid <= 9007199254740991),
      process_start_identity TEXT NOT NULL CHECK (length(process_start_identity) > 0),
      port INTEGER NOT NULL CHECK (port BETWEEN 1 AND 65535)
    )`)
    database.query(
      "INSERT INTO owner (id, pid, process_start_identity, port) VALUES (1, ?, ?, ?)",
    ).run(process.pid, await processStartIdentity(process.pid), port)
  } finally {
    database.close()
  }
}

const readStream = async (stream: ReadableStream<Uint8Array>): Promise<string> =>
  new Response(stream).text()

const runCli = async (home: string, ...args: readonly string[]): Promise<CliResult> => {
  const child = Bun.spawn([process.execPath, cliRunner, ...args], {
    cwd: resolve(import.meta.dirname, "../../.."),
    env: {
      ...process.env,
      HOME: home,
      USERPROFILE: home,
    },
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  })
  let timedOut = false
  const timeout = setTimeout(() => {
    timedOut = true
    child.kill()
  }, CLI_TIMEOUT_MS)
  const [stdout, stderr, exitCode] = await Promise.all([
    readStream(child.stdout),
    readStream(child.stderr),
    child.exited,
  ])
  clearTimeout(timeout)
  if (timedOut) {
    throw new Error([
      `CLI timed out for ${args.join(" ")}`,
      `transport trace: ${transportTrace.join(", ") || "none"}`,
      `stdout: ${JSON.stringify(stdout)}`,
      `stderr: ${JSON.stringify(stderr)}`,
    ].join("; "))
  }
  return { exitCode, stdout, stderr }
}

const success = (request: RpcRequest, value: unknown): Response => new Response(`${JSON.stringify({
  _tag: "Exit",
  requestId: request.id,
  exit: { _tag: "Success", value },
})}\n`, {
  status: 200,
  headers: {
    "connection": "close",
    "content-type": "application/ndjson",
  },
})

describe("model command JSON process boundary", () => {
  let home: string
  let residency: ModelResidency
  let requests: RecordedRpc[]

  const health = () => ({
    service: "magnitude-acn" as const,
    version: SDK_VERSION,
    revision: SDK_REVISION,
    id: AcnInstanceIdSchema.make("models-json-e2e"),
    pid: process.pid,
    state: { _tag: "Ready" as const },
  })

  const catalog = (): ModelCatalogState => {
    const model = makeModel()
    return {
      _tag: "Ready",
      providers: [],
      models: [{
        _tag: "Local",
        product: {
          ...model,
          state: { ...model.state, residencyState: residency },
        },
        offering: Option.none(),
      }],
      failures: [],
      localModelPreparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: true, settledModels: 1, totalModels: 1 },
      },
    }
  }

  const server = Bun.serve({
    port: 0,
    hostname: "127.0.0.1",
    async fetch(request) {
      const url = new URL(request.url)
      if (request.method === "GET" && url.pathname === "/health") {
        transportTrace.push("GET /health")
        return Response.json(health(), { headers: { connection: "close" } })
      }
      if (request.method !== "POST" || url.pathname !== "/rpc") {
        return new Response("Not found", { status: 404 })
      }
      const line = (await request.text()).split("\n").find((candidate) => candidate.length > 0)
      if (line === undefined) return new Response("Missing RPC request", { status: 400 })
      const rpc = JSON.parse(line) as RpcRequest
      transportTrace.push(`RPC ${rpc.tag} id=${String(rpc.id)} (${typeof rpc.id})`)
      requests.push({ tag: rpc.tag, ...(rpc.payload === undefined ? {} : { payload: rpc.payload }) })
      switch (rpc.tag) {
        case "Health": return success(rpc, health())
        case "GetModelCatalog": return success(
          rpc,
          Schema.encodeSync(ModelCatalogStateSchema)(catalog()),
        )
        case "LoadLocalModel":
          // ICN's ensureModelInstance response is returned only after the model is ready.
          residency = {
            _tag: "Ready",
            allocation: {
              contextWindowTokens: 32_768,
              parallelSequences: 1,
              physicalContextTokens: 32_768,
              memoryDomains: [],
            },
          }
          return success(rpc, {})
        case "StopActiveLocalModel":
          residency = { _tag: "Unloaded" }
          return success(rpc, {})
        default: return new Response(`Unexpected RPC ${rpc.tag}`, { status: 500 })
      }
    },
  })

  beforeAll(async () => {
    home = await mkdtemp(join(tmpdir(), "magnitude-models-json-e2e-"))
    residency = { _tag: "Unloaded" }
    requests = []
    transportTrace = []
    if (server.port === undefined) throw new Error("Isolated ACN server has no assigned port")
    await writeOwnerRecord(home, server.port)
  })

  afterAll(async () => {
    server.stop(true)
    await rm(home, { recursive: true, force: true })
  })

  it("runs successful JSON load, status, and stop transitions through the real CLI", async () => {
    const load = await runCli(home, "models", "load", TEST_MODEL_ID, "--json")
    expect(load).toEqual({
      exitCode: 0,
      stdout: `${JSON.stringify({
        schemaVersion: 1,
        command: "models.load",
        ok: true,
        data: { modelId: TEST_MODEL_ID },
      })}\n`,
      stderr: "",
    })

    const loadedStatus = await runCli(home, "models", "status", TEST_MODEL_ID, "--json")
    expect(loadedStatus.exitCode).toBe(0)
    expect(loadedStatus.stderr).toBe("")
    expect(JSON.parse(loadedStatus.stdout)).toMatchObject({
      schemaVersion: 1,
      command: "models.status",
      ok: true,
      data: {
        state: "ready",
        models: [{
          modelId: TEST_MODEL_ID,
          installation: "installed",
          residency: "ready",
        }],
      },
    })
    expect(loadedStatus.stdout.endsWith("\n")).toBe(true)
    expect(loadedStatus.stdout.split("\n")).toHaveLength(2)

    const stop = await runCli(home, "models", "stop", "--json")
    expect(stop).toEqual({
      exitCode: 0,
      stdout: `${JSON.stringify({
        schemaVersion: 1,
        command: "models.stop",
        ok: true,
        data: {},
      })}\n`,
      stderr: "",
    })

    const stoppedStatus = await runCli(home, "models", "status", TEST_MODEL_ID, "--json")
    expect(stoppedStatus.exitCode).toBe(0)
    expect(stoppedStatus.stderr).toBe("")
    expect(JSON.parse(stoppedStatus.stdout)).toMatchObject({
      data: { models: [{ residency: "unloaded" }] },
    })
  }, 30_000)

  it("keeps JSON and human mutations operationally identical", async () => {
    residency = { _tag: "Unloaded" }
    requests = []
    const jsonLoad = await runCli(home, "models", "load", TEST_MODEL_ID, "--json")
    const jsonLoadRequests = requests
    requests = []
    residency = { _tag: "Unloaded" }
    const humanLoad = await runCli(home, "models", "load", TEST_MODEL_ID)
    const humanLoadRequests = requests

    expect(jsonLoad.exitCode).toBe(0)
    expect(humanLoad).toEqual({
      exitCode: 0,
      stdout: `Loaded ${TEST_MODEL_ID}.\n`,
      stderr: "",
    })
    expect(jsonLoadRequests).toEqual(humanLoadRequests)

    requests = []
    const jsonStop = await runCli(home, "models", "stop", "--json")
    const jsonStopRequests = requests
    requests = []
    residency = { _tag: "Requested" }
    const humanStop = await runCli(home, "models", "stop")
    const humanStopRequests = requests

    expect(jsonStop.exitCode).toBe(0)
    expect(humanStop).toEqual({
      exitCode: 0,
      stdout: "Stopped the active local model.\n",
      stderr: "",
    })
    expect(jsonStopRequests).toEqual(humanStopRequests)
  }, 30_000)

  it("leaves malformed invocations and help on Commander's ordinary text path", async () => {
    const missingModel = await runCli(home, "models", "load", "--json")
    expect(missingModel.exitCode).not.toBe(0)
    expect(missingModel.stdout).toBe("")
    expect(missingModel.stderr).toContain("missing required argument 'model-id'")
    expect(() => JSON.parse(missingModel.stderr)).toThrow()

    const unknownOption = await runCli(home, "models", "stop", "--json", "--unknown")
    expect(unknownOption.exitCode).not.toBe(0)
    expect(unknownOption.stdout).toBe("")
    expect(unknownOption.stderr).toContain("unknown option '--unknown'")
    expect(() => JSON.parse(unknownOption.stderr)).toThrow()

    const help = await runCli(home, "models", "status", "--json", "--help")
    expect(help.exitCode).toBe(0)
    expect(help.stdout).toContain("Usage: magnitude models status")
    expect(help.stderr).toBe("")
  })
})
