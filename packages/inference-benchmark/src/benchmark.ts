import { Context, Data, Effect, Option } from "effect"
import { analyzeTrial } from "./analysis"
import type {
  BenchmarkResult,
  ComparisonResult,
  RequestObservation,
  TargetConfiguration,
  TrialDefinition,
  TrialObservation,
  TrialPlan,
} from "./domain"
import { startMemoryProbe } from "./memory"
import { openTarget, TargetLauncher } from "./target"
import { EndpointClient, executeRequest, type EndpointConfiguration } from "./transport"

export class EvaluationError extends Data.TaggedError("EvaluationError")<{
  readonly targetId: string
  readonly operation: string
  readonly message: string
}> {}

export interface EvaluationProgressEvent {
  readonly type: "target-ready" | "warmup-completed" | "trial-started" | "trial-completed"
  readonly targetId: string
  readonly trialId?: string
  readonly at: string
}

export class EvaluationReporter extends Context.Tag("@magnitudedev/inference-benchmark/EvaluationReporter")<
  EvaluationReporter,
  { readonly emit: (event: EvaluationProgressEvent) => Effect.Effect<void> }
>() {}

const emitProgress = (event: EvaluationProgressEvent) => Effect.serviceOption(EvaluationReporter).pipe(
  Effect.flatMap(Option.match({ onNone: () => Effect.void, onSome: (reporter) => reporter.emit(event) })),
)

const evaluationError = (targetId: string, operation: string, error: unknown) =>
  new EvaluationError({
    targetId,
    operation,
    message: error instanceof Error ? error.message : String(error),
  })

function sanitizeTarget(target: TargetConfiguration): TargetConfiguration {
  if (target.kind === "existing") {
    return { ...target, apiKey: target.apiKey ? "[redacted]" : undefined }
  }
  const env = target.env && Object.fromEntries(
    Object.entries(target.env).map(([key, value]) => [
      key,
      /(token|key|secret|password)/i.test(key) ? "[redacted]" : value,
    ]),
  )
  return { ...target, apiKey: target.apiKey ? "[redacted]" : undefined, env }
}

const runRequest = (
  endpoint: EndpointConfiguration,
  request: TrialDefinition["requests"][number],
  trialStarted: number,
): Effect.Effect<RequestObservation, never, EndpointClient> =>
  Effect.gen(function* () {
    const elapsed = performance.now() - trialStarted
    if (request.releaseOffsetMs > elapsed) {
      yield* Effect.sleep(`${request.releaseOffsetMs - elapsed} millis`)
    }
    const submittedAtMs = performance.now() - trialStarted
    const observation = yield* executeRequest(endpoint, request)
    return { ...observation, submittedAtMs }
  })

const executeTrial = (
  trial: TrialDefinition,
  endpoint: EndpointConfiguration,
  rootPid: Option.Option<number>,
  targetId: string,
): Effect.Effect<TrialObservation, EvaluationError, EndpointClient> =>
  Effect.gen(function* () {
    yield* emitProgress({ type: "trial-started", targetId, trialId: trial.id, at: new Date().toISOString() })
    const observations = new Map<string, RequestObservation>()
    const setup = trial.requests.filter((request) => request.phase === "setup")
    for (const request of setup) {
      observations.set(request.id, yield* executeRequest(endpoint, request))
    }

    const startedAt = new Date().toISOString()
    const started = performance.now()
    const memoryProbe = Option.isSome(rootPid)
      ? Option.some(yield* startMemoryProbe({ rootPid: rootPid.value }))
      : Option.none()
    const measured = trial.requests.filter((request) => request.phase !== "setup")
    const pending = new Map(measured.map((request) => [request.id, request]))

    while (pending.size > 0) {
      const ready = [...pending.values()].filter((request) =>
        request.dependsOn.every((dependency) => observations.has(dependency)))
      if (ready.length === 0) {
        return yield* new EvaluationError({
          targetId: endpoint.endpoint,
          operation: "schedule",
          message: `dependency cycle in ${trial.id}`,
        })
      }
      const completed = yield* Effect.forEach(
        ready,
        (request) => runRequest(endpoint, request, started).pipe(
          Effect.map((observation) => [request.id, observation] as const),
        ),
        { concurrency: "unbounded" },
      )
      for (const [id, observation] of completed) {
        pending.delete(id)
        observations.set(id, observation)
      }
    }

    const memory = Option.isSome(memoryProbe)
      ? Option.some(yield* memoryProbe.value.stop)
      : Option.none()
    const observation = {
      trial,
      startedAt,
      makespanMs: performance.now() - started,
      requests: measured.map((request) => observations.get(request.id)!),
      setupRequests: setup.map((request) => observations.get(request.id)!),
      memory: Option.getOrUndefined(memory),
    }
    yield* emitProgress({ type: "trial-completed", targetId, trialId: trial.id, at: new Date().toISOString() })
    return observation
  })

export const evaluate = (
  plan: TrialPlan,
  target: TargetConfiguration,
): Effect.Effect<
  BenchmarkResult,
  EvaluationError,
  EndpointClient | TargetLauncher
> =>
  Effect.scoped(Effect.gen(function* () {
    const startedAt = new Date().toISOString()
    const session = yield* openTarget(target).pipe(
      Effect.mapError((error) => evaluationError(target.id, "open-target", error)),
    )
    yield* emitProgress({ type: "target-ready", targetId: target.id, at: new Date().toISOString() })
    if (session.target.parallelSequences !== plan.servingPolicy.parallelSequences) {
      return yield* new EvaluationError({
        targetId: target.id,
        operation: "conformance",
        message: `plan requires ${plan.servingPolicy.parallelSequences} parallel sequences but target declares ${session.target.parallelSequences}`,
      })
    }

    const warmup = yield* executeRequest(session.endpoint, plan.warmup)
    if (warmup.outcome === "protocol-error" || warmup.terminal === undefined) {
      return yield* new EvaluationError({
        targetId: target.id,
        operation: "conformance",
        message: warmup.error ?? "warmup did not return conforming terminal evidence",
      })
    }
    yield* emitProgress({ type: "warmup-completed", targetId: target.id, at: new Date().toISOString() })

    const trials = yield* Effect.forEach(
      plan.trials,
      (trial) => executeTrial(trial, session.endpoint, session.rootPid, target.id),
      { concurrency: 1 },
    )
    return {
      target: sanitizeTarget(session.target),
      planDigest: plan.digest,
      startedAt,
      completedAt: new Date().toISOString(),
      trials,
      analyses: trials.map(analyzeTrial),
      warnings: [],
    }
  }))

function comparisonDifferences(
  plan: TrialPlan,
  results: readonly BenchmarkResult[],
): readonly string[] {
  const differences: string[] = []
  const capacities = new Set(results.map(({ target }) => target.parallelSequences))
  if (capacities.size !== 1) differences.push("parallel serving capacity differs")

  const artifacts = results.flatMap(({ target }) => target.artifact ? [target.artifact] : [])
  if (artifacts.length === results.length && artifacts.length > 1) {
    const reference = artifacts[0]!
    for (let index = 1; index < artifacts.length; index++) {
      const artifact = artifacts[index]!
      if (artifact.kind !== reference.kind) differences.push("model artifact formats differ")
      if (artifact.repository !== reference.repository || artifact.revision !== reference.revision) {
        differences.push("model artifact repositories or revisions differ")
      }
      if (artifact.quantization !== reference.quantization) differences.push("model artifact quantizations differ")
    }
  }

  if (results.some(({ planDigest }) => planDigest !== plan.digest)) {
    differences.push("result plan digests differ")
  }
  const [reference, ...others] = results
  if (reference) {
    const promptCounts = new Map(
      reference.trials.flatMap(({ requests }) => requests).flatMap((request) =>
        request.terminal ? [[request.requestId, request.terminal.usage.promptTokens] as const] : []),
    )
    for (const result of others) {
      let matched = 0
      const deltas: number[] = []
      for (const request of result.trials.flatMap(({ requests }) => requests)) {
        const expected = promptCounts.get(request.requestId)
        const actual = request.terminal?.usage.promptTokens
        if (expected === undefined || actual === undefined) continue
        matched++
        if (actual !== expected) deltas.push(actual - expected)
      }
      if (deltas.length > 0) {
        const minimum = Math.min(...deltas)
        const maximum = Math.max(...deltas)
        const signed = (value: number) => value > 0 ? `+${value}` : String(value)
        differences.push(
          `${result.target.id}: rendered prompt token count differs for ${deltas.length}/${matched} matched requests (delta ${signed(minimum)} to ${signed(maximum)})`,
        )
      }
    }
  }
  return [...new Set(differences)]
}

export const compare = (
  plan: TrialPlan,
  targets: readonly TargetConfiguration[],
): Effect.Effect<
  ComparisonResult,
  EvaluationError,
  EndpointClient | TargetLauncher
> =>
  Effect.gen(function* () {
    if (targets.length < 2) {
      return yield* new EvaluationError({
        targetId: "comparison",
        operation: "validate",
        message: "a comparison requires at least two targets",
      })
    }
    // Sequential target lifetimes prevent two loaded copies from contending for one device.
    const results = yield* Effect.forEach(
      targets,
      (target) => evaluate(plan, target),
      { concurrency: 1 },
    )
    const differences = comparisonDifferences(plan, results)
    return {
      comparisonKind: differences.length === 0 ? "strict" : "product",
      differences,
      plan,
      results,
    }
  })
