import { Data, Effect, Option } from "effect"
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
): Effect.Effect<TrialObservation, EvaluationError, EndpointClient> =>
  Effect.gen(function* () {
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
    return {
      trial,
      startedAt,
      makespanMs: performance.now() - started,
      requests: measured.map((request) => observations.get(request.id)!),
      setupRequests: setup.map((request) => observations.get(request.id)!),
      memory: Option.getOrUndefined(memory),
    }
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

    const trials = yield* Effect.forEach(
      plan.trials,
      (trial) => executeTrial(trial, session.endpoint, session.rootPid),
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
