import * as FileSystem from "@effect/platform/FileSystem"
import * as CommandExecutor from "@effect/platform/CommandExecutor"
import { Data, Effect, Schema } from "effect"
import { appendFile, open, readFile, readdir, unlink } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { compare, evaluate, EvaluationReporter } from "./benchmark"
import { prepareCorpus } from "./corpus"
import type { ComparisonResult, ExperimentRunResult, TargetConfiguration, TrialPlan } from "./domain"
import { loadExperiment, resolveExecutionOrder, resolveExperimentPaths, type ExperimentDefinition } from "./experiment"
import { digestObject } from "./hash"
import { compileTrialPlan } from "./plan"
import { loadPreparedExperiment, type PreparedArtifact, type PreparedExperiment } from "./preparation"
import { renderComparisonMarkdown } from "./report"
import { managedIcnTarget } from "./target"

export interface RunManifest {
  readonly version: 1
  readonly runId: string
  readonly pid: number
  readonly state: "running"
  readonly startedAt: string
  readonly runDirectory: string
  readonly prepared: PreparedExperiment
  readonly plan: TrialPlan
  readonly executionOrder: readonly (readonly string[])[]
}

export interface RunEvent {
  readonly type: string
  readonly at: string
  readonly runId: string
  readonly block?: number
  readonly variantId?: string
  readonly trialId?: string
  readonly message?: string
}

export class RunError extends Data.TaggedError("RunError")<{
  readonly operation: string
  readonly message: string
}> {}

const resultsRoot = () => resolve("benchmark-results", "runs")
const lockPath = () => resolve("benchmark-results", "run.lock")
export const activeRunLockPath = lockPath

const writeJson = (path: string, value: unknown) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  yield* fs.makeDirectory(dirname(path), { recursive: true })
  const encoded = yield* Schema.encode(Schema.parseJson(Schema.Unknown, { space: 2 }))(value)
  yield* fs.writeFileString(path, `${encoded}\n`)
})

const appendEvent = (path: string, event: RunEvent) => Effect.tryPromise({
  try: () => appendFile(path, `${JSON.stringify(event)}\n`),
  catch: (error) => new RunError({ operation: "append-event", message: error instanceof Error ? error.message : String(error) }),
})

function runId(): string {
  return `${new Date().toISOString().replaceAll(":", "").replaceAll(".", "-")}-${crypto.randomUUID().slice(0, 8)}`
}

function artifactFor(prepared: PreparedExperiment, variantId: string, role: PreparedArtifact["role"]): PreparedArtifact {
  const artifact = prepared.artifacts.find((candidate) => candidate.variantId === variantId && candidate.role === role)
  if (!artifact) throw new RunError({ operation: "resolve-target", message: `missing prepared ${role} artifact for ${variantId}` })
  return artifact
}

function engineFor(prepared: PreparedExperiment, variantId: string) {
  const engine = prepared.engines.find((candidate) => candidate.variantId === variantId)
  if (!engine) throw new RunError({ operation: "resolve-target", message: `missing prepared engine for ${variantId}` })
  return engine
}

export function targetFor(prepared: PreparedExperiment, variantId: string, port: number, logPath: string): TargetConfiguration {
  const variant = prepared.experiment.variants.find((candidate) => candidate.id === variantId)
  if (!variant) throw new RunError({ operation: "resolve-target", message: `unknown variant ${variantId}` })
  const artifact = artifactFor(prepared, variantId, "target")
  const preparedEngine = engineFor(prepared, variantId)
  if (preparedEngine.kind !== variant.engine.kind) {
    throw new RunError({ operation: "resolve-target", message: `prepared engine for ${variantId} does not match the experiment` })
  }
  const artifactIdentity = {
    kind: artifact.kind,
    repository: artifact.repository,
    revision: artifact.revision,
    quantization: artifact.quantization,
  }
  const common = {
    kind: "managed" as const,
    id: variant.id,
    endpoint: `http://127.0.0.1:${port}`,
    servedModel: variant.artifact.modelId,
    readinessPath: "/health",
    parallelSequences: prepared.experiment.requestPolicy.parallelSequences,
    requestTimeoutMs: prepared.experiment.requestPolicy.requestTimeoutMs,
    logPath,
    artifact: artifactIdentity,
  }
  if (variant.engine.kind === "existing-endpoint" && preparedEngine.kind === "existing-endpoint") {
    const apiKey = variant.engine.authentication.kind === "bearer-env"
      ? process.env[variant.engine.authentication.variable]
      : undefined
    if (variant.engine.authentication.kind === "bearer-env" && !apiKey) {
      throw new RunError({
        operation: "resolve-target",
        message: `existing endpoint ${variantId} requires ${variant.engine.authentication.variable}`,
      })
    }
    return {
      kind: "existing",
      id: variant.id,
      endpoint: preparedEngine.endpoint,
      servedModel: variant.artifact.modelId,
      parallelSequences: prepared.experiment.requestPolicy.parallelSequences,
      apiKey,
      requestBody: variant.engine.requestBody,
      requestTimeoutMs: prepared.experiment.requestPolicy.requestTimeoutMs,
      artifact: artifactIdentity,
    }
  }
  if (variant.engine.kind === "icn" && preparedEngine.kind === "icn") {
    const target = managedIcnTarget({
      model: {
        ...prepared.planModel,
        artifactPath: artifact.path,
        artifactSha256: artifact.digest,
      },
      icnExecutable: preparedEngine.executable,
      port,
      maxSequences: prepared.experiment.requestPolicy.parallelSequences,
      ...(variant.engine.speculativeDecoding.kind === "dflash"
        ? {
            speculative: {
              method: "dflash" as const,
              draftSha256: artifactFor(prepared, variantId, "drafter").digest,
            },
          }
        : {}),
    })
    return { ...target, ...common }
  }
  if (variant.engine.kind === "llama.cpp") {
    const sharedContextTokens = prepared.experiment.requestPolicy.contextTokensPerSequence
      * prepared.experiment.requestPolicy.parallelSequences
    const args = [
      "--host", "127.0.0.1", "--port", String(port), "--model", artifact.path,
      "--alias", variant.artifact.modelId, "--ctx-size", String(sharedContextTokens),
      "--parallel", String(prepared.experiment.requestPolicy.parallelSequences), "--jinja",
      "--cache-type-k", "f16", "--cache-type-v", "f16",
    ]
    if (variant.engine.continuousBatching) args.push("--cont-batching")
    if (variant.engine.flashAttention) args.push("--flash-attn", "on")
    if (variant.engine.speculativeDecoding.kind === "mtp") {
      const drafter = artifactFor(prepared, variantId, "drafter")
      args.push(
        "--spec-type", "draft-mtp",
        "--spec-draft-model", drafter.path,
        "--spec-draft-n-max", String(variant.engine.speculativeDecoding.maxDraftTokens),
      )
    }
    if (preparedEngine.kind !== "llama.cpp") throw new RunError({ operation: "resolve-target", message: `invalid prepared llama.cpp engine for ${variantId}` })
    return { ...common, engine: "llama.cpp", executable: preparedEngine.executable, args }
  }
  if (variant.engine.kind === "mlx-vlm" && preparedEngine.kind === "mlx-vlm") {
    const drafter = artifactFor(prepared, variantId, "drafter")
    return {
      ...common,
      engine: "mlx-vlm",
      servedModel: artifact.path,
      executable: "uv",
      args: [
        "run", "--frozen", "--no-sync", "--project", variant.engine.pythonProject,
        "mlx_vlm.server", "--model", artifact.path,
        "--draft-model", drafter.path, "--draft-kind", "mtp",
        "--draft-block-size", String(variant.engine.speculativeDecoding.maxDraftTokens + 1),
        "--host", "127.0.0.1", "--port", String(port),
        "--prefill-step-size", String(variant.engine.prefillStepSize),
        "--max-num-seqs", String(prepared.experiment.requestPolicy.parallelSequences),
        "--log-progress-interval", "0",
      ],
    }
  }
  if (variant.engine.kind !== "mlx-lm" || preparedEngine.kind !== "mlx-lm") {
    throw new RunError({ operation: "resolve-target", message: `unsupported engine for ${variantId}` })
  }
  if (!artifact.manifestPath) throw new RunError({ operation: "resolve-target", message: `MLX artifact ${variantId} has no manifest` })
  return {
    ...common,
    engine: "mlx-lm",
    executable: "uv",
    args: [
      "run", "--frozen", "--no-sync", "--project", variant.engine.pythonProject,
      "magnitude-mlx-benchmark-server", "--model", artifact.path,
      "--served-model", variant.artifact.modelId, "--model-revision", artifact.revision,
      "--artifact-manifest", artifact.manifestPath, "--host", "127.0.0.1", "--port", String(port),
      "--prefill-step-size", String(variant.engine.prefillStepSize),
      "--prompt-cache-entries", String(variant.engine.promptCacheEntries),
    ],
  }
}

export function assertSpeculativeEvidence(experiment: ExperimentDefinition, comparison: ComparisonResult): void {
  const speculativeVariants = new Map<string, "mtp" | "dflash">()
  for (const variant of experiment.variants) {
    if ((variant.engine.kind === "llama.cpp" || variant.engine.kind === "mlx-vlm")
      && variant.engine.speculativeDecoding.kind === "mtp") {
      speculativeVariants.set(variant.id, "mtp")
    }
    if (variant.engine.kind === "icn" && variant.engine.speculativeDecoding.kind === "dflash") {
      speculativeVariants.set(variant.id, "dflash")
    }
  }
  for (const result of comparison.results) {
    const method = speculativeVariants.get(result.target.id)
    if (!method) continue
    const requests = result.trials.flatMap((trial) => [
      ...(trial.setupRequests ?? []),
      ...trial.requests,
    ])
    let drafted = 0
    let accepted = 0
    let distributionBacked = 0
    for (const request of requests) {
      const draftTokens = request.terminal?.timings.draftTokens
      const acceptedDraftTokens = request.terminal?.timings.acceptedDraftTokens
      if (draftTokens === undefined || acceptedDraftTokens === undefined) {
        throw new RunError({
          operation: "validate-speculative-evidence",
          message: `${result.target.id} request ${request.requestId} did not return native ${method} draft counters`,
        })
      }
      const proposalDistributionDraftTokens = request.terminal?.timings.proposalDistributionDraftTokens ?? 0
      if (!Number.isSafeInteger(draftTokens) || !Number.isSafeInteger(acceptedDraftTokens)
        || !Number.isSafeInteger(proposalDistributionDraftTokens)
        || draftTokens < 0 || acceptedDraftTokens < 0 || acceptedDraftTokens > draftTokens
        || proposalDistributionDraftTokens < 0 || proposalDistributionDraftTokens > draftTokens) {
        throw new RunError({
          operation: "validate-speculative-evidence",
          message: `${result.target.id} request ${request.requestId} returned inconsistent ${method} draft counters`,
        })
      }
      drafted += draftTokens
      accepted += acceptedDraftTokens
      distributionBacked += proposalDistributionDraftTokens
    }
    if (requests.length === 0 || drafted === 0) {
      throw new RunError({
        operation: "validate-speculative-evidence",
        message: `${result.target.id} did not demonstrate active ${method} drafting`,
      })
    }
    if (method === "dflash") {
      if (accepted === 0) {
        throw new RunError({
          operation: "validate-speculative-evidence",
          message: `${result.target.id} accepted no DFlash draft tokens`,
        })
      }
      // Under stochastic sampling the DFlash2 drafter emits a proposal distribution
      // for every draft; zero distribution-backed drafts means the run exercised
      // only the greedy path and cannot certify distribution-aware verification.
      if (experiment.requestPolicy.temperature > 0 && distributionBacked === 0) {
        throw new RunError({
          operation: "validate-speculative-evidence",
          message: `${result.target.id} produced no proposal-distribution-backed drafts at temperature ${experiment.requestPolicy.temperature}`,
        })
      }
    }
  }
}

const withRunLock = <A, E, R>(id: string, effect: Effect.Effect<A, E, R>): Effect.Effect<A, E | RunError, R | FileSystem.FileSystem> =>
  Effect.acquireUseRelease(
    Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      yield* fs.makeDirectory(dirname(lockPath()), { recursive: true }).pipe(
        Effect.mapError((error) => new RunError({ operation: "acquire-lock", message: String(error) })),
      )
      return yield* Effect.tryPromise({
        try: async () => {
          let handle
          try {
            handle = await open(lockPath(), "wx")
          } catch (error) {
            const code = error && typeof error === "object" && "code" in error ? Reflect.get(error, "code") : undefined
            if (code !== "EEXIST") throw error
            const current = JSON.parse(await readFile(lockPath(), "utf8")) as { pid?: unknown }
            if (Number.isSafeInteger(current.pid)) {
              try {
                process.kill(Number(current.pid), 0)
                throw new RunError({ operation: "acquire-lock", message: `benchmark run PID ${current.pid} is active` })
              } catch (probeError) {
                if (probeError instanceof RunError) throw probeError
              }
            }
            await unlink(lockPath())
            handle = await open(lockPath(), "wx")
          }
          await handle.writeFile(JSON.stringify({ runId: id, pid: process.pid, startedAt: new Date().toISOString() }))
          return handle
        },
        catch: (error) => new RunError({ operation: "acquire-lock", message: `another benchmark run is active (${error instanceof Error ? error.message : String(error)})` }),
      })
    }),
    () => effect,
    (handle) => Effect.promise(async () => { await handle.close(); await unlink(lockPath()).catch(() => undefined) }),
  )

export const runExperiment = (experimentPath: string): Effect.Effect<ExperimentRunResult, RunError, FileSystem.FileSystem | CommandExecutor.CommandExecutor | import("./transport").EndpointClient | import("./target").TargetLauncher> =>
  Effect.gen(function* () {
    const loaded = yield* loadExperiment(experimentPath).pipe(Effect.mapError((error) => new RunError({ operation: error.operation, message: error.message })))
    const resolved = resolveExperimentPaths(loaded, experimentPath)
    const prepared = yield* loadPreparedExperiment(resolved.id).pipe(Effect.mapError((error) => new RunError({ operation: error.operation, message: error.message })))
    if (digestObject(prepared.experiment) !== digestObject(resolved)) {
      return yield* new RunError({ operation: "validate-prepared", message: "experiment changed after preparation; run prepare again" })
    }
    const corpus = yield* prepareCorpus({ root: prepared.corpusRoot, offline: true }).pipe(Effect.mapError((error) => new RunError({ operation: "load-corpus", message: String(error) })))
    const plan = yield* compileTrialPlan(corpus, prepared.planModel, {
      ...(prepared.experiment.suite.kind === "agent-core"
        ? { profile: prepared.experiment.suite.profile }
        : { contextSweep: prepared.experiment.suite }),
      maxContextTokens: prepared.experiment.requestPolicy.contextTokensPerSequence,
      parallelSequences: prepared.experiment.requestPolicy.parallelSequences,
      maxOutputTokens: prepared.experiment.requestPolicy.maxOutputTokens,
      temperature: prepared.experiment.requestPolicy.temperature,
      topP: prepared.experiment.requestPolicy.topP,
      seed: prepared.experiment.requestPolicy.seed,
      enableThinking: prepared.experiment.requestPolicy.enableThinking,
    }).pipe(Effect.mapError((error) => new RunError({ operation: "compile-plan", message: error.message })))
    const id = runId()
    const directory = join(resultsRoot(), id)
    const logs = join(directory, "logs")
    const eventsPath = join(directory, "events.jsonl")
    const order = resolveExecutionOrder(prepared.experiment)
    const startedAt = new Date().toISOString()
    const manifest: RunManifest = {
      version: 1, runId: id, pid: process.pid, state: "running", startedAt,
      runDirectory: directory, prepared, plan, executionOrder: order,
    }
    const fs = yield* FileSystem.FileSystem

    const execute = Effect.gen(function* () {
      yield* appendEvent(eventsPath, { type: "run-started", at: startedAt, runId: id })
      let activeBlock = 0
      const reporter = EvaluationReporter.of({
        emit: (event) => appendEvent(eventsPath, {
          type: event.type, at: event.at, runId: id, block: activeBlock,
          variantId: event.targetId, trialId: event.trialId,
        }).pipe(Effect.ignore),
      })
      const blocks = []
      for (let block = 0; block < order.length; block++) {
        activeBlock = block
        const variantOrder = order[block]!
        yield* appendEvent(eventsPath, { type: "block-started", at: new Date().toISOString(), runId: id, block })
        const targets = variantOrder.map((variantId) => targetFor(prepared, variantId, 8091, join(logs, `${variantId}.block-${block}.log`)))
        const executeTargets = targets.length === 1
          ? evaluate(plan, targets[0]!).pipe(Effect.map((result) => ({
              comparisonKind: "strict" as const,
              differences: [] as const,
              plan,
              results: [result],
            })))
          : compare(plan, targets)
        const comparison = yield* executeTargets.pipe(
          Effect.provideService(EvaluationReporter, reporter),
          Effect.mapError((error) => new RunError({ operation: error.operation, message: error.message })),
        )
        assertSpeculativeEvidence(prepared.experiment, comparison)
        blocks.push({ index: block, variantOrder, comparison })
        yield* appendEvent(eventsPath, { type: "block-completed", at: new Date().toISOString(), runId: id, block })
      }
      const completedAt = new Date().toISOString()
      const result: ExperimentRunResult = { runId: id, experimentId: prepared.experiment.id, startedAt, completedAt, blocks }
      yield* writeJson(join(directory, "result.json"), result)
      yield* fs.writeFileString(join(directory, "report.md"), blocks.map(({ index, comparison }) => `# Block ${index}\n\n${renderComparisonMarkdown(comparison)}`).join("\n\n"))
      yield* appendEvent(eventsPath, { type: "run-completed", at: completedAt, runId: id })
      return result
    }).pipe(
      Effect.tapError((error) => appendEvent(eventsPath, {
        type: "run-failed", at: new Date().toISOString(), runId: id, message: error instanceof Error ? error.message : String(error),
      }).pipe(Effect.ignore)),
      Effect.onInterrupt(() => appendEvent(eventsPath, {
        type: "run-cancelled", at: new Date().toISOString(), runId: id,
      }).pipe(Effect.ignore)),
    )
    return yield* withRunLock(id, Effect.gen(function* () {
      yield* fs.makeDirectory(logs, { recursive: true })
      yield* writeJson(join(directory, "manifest.json"), manifest)
      yield* fs.writeFileString(eventsPath, "")
      return yield* execute
    }))
  }).pipe(Effect.mapError((error) => error instanceof RunError ? error : new RunError({
    operation: "run", message: error instanceof Error ? error.message : String(error),
  })))

export interface RunSummary {
  readonly runId: string
  readonly state: "running" | "completed" | "failed"
  readonly startedAt: string
  readonly experimentId: string
  readonly directory: string
}

export const listRuns = (): Effect.Effect<readonly RunSummary[], RunError> => Effect.tryPromise({
  try: async () => {
    const entries = await readdir(resultsRoot(), { withFileTypes: true }).catch(() => [])
    const candidates = await Promise.all(entries.filter((entry) => entry.isDirectory()).map(async (entry): Promise<RunSummary | undefined> => {
      try {
        const directory = join(resultsRoot(), entry.name)
        const manifest = JSON.parse(await readFile(join(directory, "manifest.json"), "utf8")) as RunManifest
        let state: RunSummary["state"] = "running"
        try { await readFile(join(directory, "result.json")); state = "completed" } catch { /* active or failed */ }
        if (state === "running") {
          const events = await readFile(join(directory, "events.jsonl"), "utf8").catch(() => "")
          if (/"type":"run-(failed|cancelled)"/.test(events)) state = "failed"
          else {
            try { process.kill(manifest.pid, 0) } catch { state = "failed" }
          }
        }
        return { runId: entry.name, state, startedAt: manifest.startedAt, experimentId: manifest.prepared.experiment.id, directory }
      } catch {
        return undefined
      }
    }))
    return candidates.filter((run): run is RunSummary => run !== undefined)
      .sort((left, right) => right.startedAt.localeCompare(left.startedAt))
  },
  catch: (error) => new RunError({ operation: "list-runs", message: error instanceof Error ? error.message : String(error) }),
})

export const runDirectory = (id: string) => join(resultsRoot(), id)
