import { BunContext } from "@effect/platform-bun"
import * as FileSystem from "@effect/platform/FileSystem"
import {
  DashboardExperiment,
  DashboardRun,
  DashboardRunDetail,
  activeRunLockPath,
  analyzeTrial,
  digestObject,
  discoverExperiments,
  listRuns,
  loadExperiment,
  loadPreparedExperiment,
  resolveExperimentPaths,
  runDirectory,
  type TrialObservation,
} from "@magnitudedev/inference-benchmark"
import { Effect, Schema } from "effect"
import { readFile } from "node:fs/promises"
import { join, resolve } from "node:path"

const port = Number(process.env.INFERENCE_BENCHMARK_DASHBOARD_PORT ?? 4897)
const workspace = resolve(import.meta.dir, "../../..")
process.chdir(workspace)
const cors = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "content-type",
}

const runEffect = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem>) =>
  Effect.runPromise(effect.pipe(Effect.provide(BunContext.layer)))

function json(value: unknown, init: ResponseInit = {}): Response {
  return Response.json(value, { ...init, headers: { ...cors, ...(init.headers ?? {}) } })
}

async function experiments(): Promise<readonly DashboardExperiment[]> {
  const discovered = await runEffect(discoverExperiments())
  return Promise.all(discovered.map(async ({ path }) => {
    const experiment = await runEffect(loadExperiment(path))
    const prepared = await runEffect(loadPreparedExperiment(experiment.id)).catch(() => undefined)
    const resolved = resolveExperimentPaths(experiment, path)
    return {
      id: experiment.id,
      title: experiment.title,
      path,
      profile: experiment.suite.kind === "agent-core" ? experiment.suite.profile : "context-sweep",
      prepared: prepared !== undefined && digestObject(prepared.experiment) === digestObject(resolved),
      requestPolicy: experiment.requestPolicy,
      execution: experiment.execution,
      variants: experiment.variants.map((variant) => ({
        id: variant.id,
        engine: variant.engine.kind,
        artifact: {
          kind: variant.artifact.kind,
          repository: variant.artifact.repository,
          revision: variant.artifact.revision,
          quantization: variant.artifact.kind === "gguf"
            ? variant.artifact.quantization.scheme
            : variant.artifact.quantization.family === "mlx-affine"
              ? `${variant.artifact.quantization.bits}-bit/group-${variant.artifact.quantization.groupSize}`
              : variant.artifact.quantization.dtype,
        },
      })),
    }
  }))
}

async function experimentById(id: string): Promise<DashboardExperiment | undefined> {
  return (await experiments()).find((experiment) => experiment.id === id)
}

function spawnCli(args: readonly string[]): number {
  const process = Bun.spawn(["bun", "run", "benchmark", ...args], {
    cwd: workspace,
    stdin: "ignore",
    stdout: "inherit",
    stderr: "inherit",
  })
  process.unref()
  return process.pid
}

async function runDetail(id: string): Promise<DashboardRunDetail | undefined> {
  const summary = (await Effect.runPromise(listRuns())).find((run) => run.runId === id)
  if (!summary) return undefined
  const directory = runDirectory(id)
  const manifest = JSON.parse(await readFile(join(directory, "manifest.json"), "utf8"))
  const eventText = await readFile(join(directory, "events.jsonl"), "utf8").catch(() => "")
  const events = eventText.trim() ? eventText.trimEnd().split("\n").map((line) => JSON.parse(line)) : []
  const result = await readFile(join(directory, "result.json"), "utf8").then(JSON.parse).then(reanalyze).catch(() => null)
  return { run: summary, manifest, result, events }
}

function reanalyze(value: unknown): unknown {
  const result = value as { blocks?: Array<{ comparison?: { results?: Array<{ trials?: TrialObservation[]; analyses?: unknown }> } }> }
  for (const block of result.blocks ?? []) {
    for (const evaluation of block.comparison?.results ?? []) {
      if (evaluation.trials) evaluation.analyses = evaluation.trials.map(analyzeTrial)
    }
  }
  return value
}

Bun.serve({
  hostname: "127.0.0.1",
  port,
  async fetch(request) {
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors })
    try {
      const url = new URL(request.url)
      const parts = url.pathname.split("/").filter(Boolean)
      if (request.method === "GET" && url.pathname === "/api/experiments") {
        return json(Schema.encodeUnknownSync(Schema.Array(DashboardExperiment))(await experiments()))
      }
      if (parts[0] === "api" && parts[1] === "experiments" && parts[2]) {
        const experiment = await experimentById(decodeURIComponent(parts[2]))
        if (!experiment) return json({ error: "experiment not found" }, { status: 404 })
        if (request.method === "GET" && parts.length === 3) return json(Schema.encodeUnknownSync(DashboardExperiment)(experiment))
        if (request.method === "POST" && (parts[3] === "prepare" || parts[3] === "runs")) {
          const command = parts[3] === "prepare" ? "prepare" : "run"
          return json({ pid: spawnCli([command, experiment.path]) }, { status: 202 })
        }
      }
      if (request.method === "GET" && url.pathname === "/api/runs") {
        return json(Schema.encodeUnknownSync(Schema.Array(DashboardRun))(await Effect.runPromise(listRuns())))
      }
      if (parts[0] === "api" && parts[1] === "runs" && parts[2]) {
        const id = decodeURIComponent(parts[2])
        if (request.method === "GET" && parts.length === 3) {
          const detail = await runDetail(id)
          return detail ? json(Schema.encodeUnknownSync(DashboardRunDetail)(detail)) : json({ error: "run not found" }, { status: 404 })
        }
        if (request.method === "GET" && parts[3] === "events") {
          const detail = await runDetail(id)
          const after = Number(url.searchParams.get("after") ?? 0)
          return detail ? json(detail.events.slice(Number.isSafeInteger(after) ? after : 0)) : json({ error: "run not found" }, { status: 404 })
        }
        if (request.method === "GET" && parts[3] === "result") {
          const detail = await runDetail(id)
          return detail?.result ? json(detail.result) : json({ error: "result not available" }, { status: 404 })
        }
        if (request.method === "POST" && parts[3] === "cancel") {
          const manifest = JSON.parse(await readFile(join(runDirectory(id), "manifest.json"), "utf8")) as { pid?: unknown; runId?: unknown }
          const lock = JSON.parse(await readFile(activeRunLockPath(), "utf8")) as { pid?: unknown; runId?: unknown }
          if (manifest.runId !== id || lock.runId !== id || !Number.isSafeInteger(manifest.pid) || lock.pid !== manifest.pid) return json({ error: "run is not the active benchmark process" }, { status: 409 })
          process.kill(Number(manifest.pid), "SIGINT")
          return json({ cancelled: true }, { status: 202 })
        }
      }
      return json({ error: "not found" }, { status: 404 })
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 })
    }
  },
})

console.log(`Inference benchmark dashboard API: http://127.0.0.1:${port}`)
