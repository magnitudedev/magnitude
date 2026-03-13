/**
 * Generic eval runner — executes evals against models,
 * then runs scenario checks against raw responses.
 */

import { Effect, Layer, Stream } from 'effect'
import { type ChatMessage, BamlClientHttpError, BamlValidationError } from '@magnitudedev/llm-core'
import { ModelResolver, makeModelResolver, makeNoopTracer, CodingAgentChat } from '@magnitudedev/providers'
import { isRunnableEval, type Eval, type ModelSpec, type ScenarioResult, type EvalRunResult, type Scenario, type CheckResult } from './types'
import type { TestSandboxResult } from './test-sandbox'
import { generateModelReport, generateSummaryReport } from './results'
import { getEvalProviderClient } from './provider-runtime'

function resolveCheckScore(checkResult: CheckResult): number {
  return checkResult.score ?? (checkResult.passed ? 1 : 0)
}

function computeScenarioScore(checks: Record<string, CheckResult>): number {
  const checkResults = Object.values(checks)
  if (checkResults.length === 0) return 0
  const total = checkResults.reduce((sum, checkResult) => sum + resolveCheckScore(checkResult), 0)
  return total / checkResults.length
}

/**
 * Evaluate a single scenario's response by running checks against the raw response.
 * Note: The old js-act sandbox path has been removed. Checks now operate on raw response text.
 */
export async function evaluateScenarioResponse(raw: string, scenario: Scenario): Promise<ScenarioResult> {
  const checks: Record<string, CheckResult> = {}
  let allPassed = true

  const emptySandboxResult: TestSandboxResult = {
    calls: [],
    events: [],
  }
  for (const check of scenario.checks) {
    const checkResult = check.evaluate(raw, emptySandboxResult)
    checks[check.id] = checkResult
    if (!checkResult.passed) allPassed = false
  }

  return {
    scenarioId: scenario.id,
    checks,
    passed: allPassed,
    score: computeScenarioScore(checks),
    rawResponse: raw
  }
}

/**
 * Check if an error is retryable (rate limit, timeout, server error).
 * Non-retryable: 4xx (except 408/429), validation errors.
 */
function isRetryableError(error: unknown): boolean {
  if (error instanceof BamlClientHttpError) {
    const status = error.status_code
    if (status !== undefined && status >= 400 && status < 500) {
      // 429 (rate limit) and 408 (timeout) are retryable
      return status === 429 || status === 408
    }
    // 5xx and unknown status codes are retryable
    return true
  }
  if (error instanceof BamlValidationError) {
    return false
  }
  // Unknown errors — retryable (network failures, etc.)
  return true
}

const MAX_RETRIES = 6
const BASE_DELAY_MS = 1000
const BACKOFF_MULTIPLIER = 1.5

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

/**
 * Call a model for a scenario using the correct provider path.
 * Uses the same routing as the main agent (Codex, Copilot, BAML).
 * Retries retryable errors (rate limits, timeouts, server errors) with exponential backoff.
 */
export async function callModel(
  systemPrompt: string,
  messages: ChatMessage[],
  modelSpec: ModelSpec
): Promise<string> {
  const client = await getEvalProviderClient()
  const auth = await client.auth.getAuth(modelSpec.provider)
  await client.state.setSelection('primary', modelSpec.provider, modelSpec.model, auth ?? null, { persist: false })

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const chatStream = await Effect.runPromise(
        Effect.gen(function* () {
          const runtime = yield* ModelResolver
          const model = yield* runtime.resolve('primary')
          return yield* model.invoke(
            CodingAgentChat,
            {
              systemPrompt,
              messages,
              ackTurn: '<lenses>task: no</lenses>\n<comms>\n<message>Ready.</message>\n</comms>',
            },
          )
        }).pipe(Effect.provide(Layer.merge(makeModelResolver().pipe(Layer.provide(client.layer), Layer.provide(makeNoopTracer())), makeNoopTracer()))),
      )
      const result = await Effect.runPromise(Stream.runFold(chatStream.stream, '', (acc, chunk) => acc + chunk))
      return result
    } catch (error) {
      if (attempt < MAX_RETRIES && isRetryableError(error)) {
        const delay = BASE_DELAY_MS * Math.pow(BACKOFF_MULTIPLIER, attempt) * (0.5 + Math.random() * 0.5)
        const status = error instanceof BamlClientHttpError ? ` (${error.status_code})` : ''
        console.warn(`  [retry ${attempt + 1}/${MAX_RETRIES}] ${modelSpec.provider}:${modelSpec.model}${status} — retrying in ${Math.round(delay)}ms`)
        await sleep(delay)
        continue
      }
      throw error
    }
  }

  throw new Error('Unreachable')
}

/**
 * Callbacks for progressive result reporting
 */
export interface RunCallbacks {
  onScenarioStart?: (scenario: Scenario, index: number, total: number) => void
  onScenarioComplete?: (scenario: Scenario, result: ScenarioResult, index: number, total: number) => void
  onModelStart?: (modelSpec: ModelSpec) => void
  onModelComplete?: (modelSpec: ModelSpec, result: EvalRunResult) => void
}

/**
 * Run a single scenario against a model (LLM call + sandbox evaluation)
 */
async function runScenario(
  scenario: Scenario,
  modelSpec: ModelSpec
): Promise<ScenarioResult> {
  const raw = await callModel('', scenario.messages, modelSpec)
  return evaluateScenarioResponse(raw, scenario)
}

/**
 * Run a full eval against a single model — scenarios run in parallel
 */
export async function runEval(
  eval_: Eval,
  modelSpec: ModelSpec,
  options?: {
    scenarios?: string[]
    callbacks?: RunCallbacks
    concurrency?: number
    repeat?: number
    resultsDir?: string
  }
): Promise<EvalRunResult> {
  const selected = options?.scenarios ?? []
  const scenarios = selected.length > 0
    ? eval_.scenarios.filter(s => selected.includes(s.id))
    : eval_.scenarios

  const cb = options?.callbacks
  const concurrency = options?.concurrency ?? 4
  const repeat = options?.repeat ?? 1
  const resultsDir = options?.resultsDir
  const filePrefix = resultsDir
    ? `${resultsDir}/${`${modelSpec.provider}-${modelSpec.model}`.replace(/[/:]/g, '-')}`
    : null

  cb?.onModelStart?.(modelSpec)

  // Expand scenarios for repeat runs
  const expandedScenarios: { scenario: Scenario; runIndex: number }[] = []
  for (const scenario of scenarios) {
    for (let r = 0; r < repeat; r++) {
      expandedScenarios.push({ scenario, runIndex: r })
    }
  }

  // Run scenarios in parallel with concurrency limit
  const results: ScenarioResult[] = new Array(expandedScenarios.length)
  let nextIndex = 0
  let completedCount = 0

  function buildPartialResult(): EvalRunResult {
    const completed = results.filter(r => r != null)
    const averageScore = completed.length > 0
      ? completed.reduce((sum, scenarioResult) => sum + scenarioResult.score, 0) / completed.length
      : 0

    return {
      evalId: eval_.id,
      model: modelSpec.model,
      provider: modelSpec.provider,
      scenarios: completed,
      passedCount: completed.filter(r => r.passed).length,
      totalCount: expandedScenarios.length,
      averageScore,
    }
  }

  function flushResults(force = false) {
    if (!filePrefix) return
    const partial = buildPartialResult()
    Bun.write(`${filePrefix}.json`, JSON.stringify(partial, null, 2))
    if (force || completedCount % 10 === 0) {
      Bun.write(`${filePrefix}.md`, generateModelReport(partial, scenarios))
      Bun.write(`${resultsDir}/summary.md`, generateSummaryReport([partial]))
    }
  }

  async function worker() {
    while (nextIndex < expandedScenarios.length) {
      const i = nextIndex++
      const { scenario, runIndex } = expandedScenarios[i]
      cb?.onScenarioStart?.(scenario, i, expandedScenarios.length)

      try {
        const result = isRunnableEval(eval_)
          ? await eval_.runScenario(scenario, modelSpec)
          : await runScenario(scenario, modelSpec)
        if (repeat > 1) result.runIndex = runIndex
        results[i] = result
        completedCount++
        cb?.onScenarioComplete?.(scenario, result, i, expandedScenarios.length)
        flushResults()
      } catch (error) {
        const checks: Record<string, CheckResult> = {}
        for (const check of scenario.checks) {
          checks[check.id] = {
            passed: false,
            message: 'Model call failed: ' + (error instanceof Error ? error.message : String(error))
          }
        }
        const failResult: ScenarioResult = {
          scenarioId: scenario.id,
          checks,
          passed: false,
          score: 0,
          rawResponse: ''
        }
        results[i] = failResult
        completedCount++
        cb?.onScenarioComplete?.(scenario, failResult, i, expandedScenarios.length)
        flushResults()
      }
    }
  }

  // Spawn workers up to concurrency limit
  const workers = Array.from(
    { length: Math.min(concurrency, expandedScenarios.length) },
    () => worker()
  )
  await Promise.all(workers)

  const passedCount = results.filter(r => r.passed).length
  const averageScore = results.length > 0
    ? results.reduce((sum, scenarioResult) => sum + scenarioResult.score, 0) / results.length
    : 0

  const evalResult: EvalRunResult = {
    evalId: eval_.id,
    model: modelSpec.model,
    provider: modelSpec.provider,
    scenarios: results,
    passedCount,
    totalCount: results.length,
    averageScore,
  }

  cb?.onModelComplete?.(modelSpec, evalResult)

  // Final flush (force markdown regeneration)
  flushResults(true)

  return evalResult
}

