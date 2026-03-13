import { Effect, Layer } from 'effect'
import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, CheckResult } from '../../types'
import { ModelResolver, makeModelResolver, makeNoopTracer, ExtractMemoryDiff } from '@magnitudedev/providers'
import { getEvalProviderClient } from '../../provider-runtime'

import { applyMemoryDiff } from '../../../../packages/agent/src/memory/memory-file'
import { ALL_SCENARIOS, VARIANTS } from './scenarios'
import type { MemoryEvalScenario, MemorySingleScenario, MemoryMultiScenario, MemoryDiffResult } from './types'
import {
  validJsonObject,
  validCategories,
  imperativeLineShape,
  exactEmpty,
  operationBounds,
  requiredCategories,
  allowedCategories,
  duplicateDetection,
  hasUpdateOrDeletion,
} from './checks'
import { runJudgeChecks } from './judge'



function computeScore(checks: Record<string, CheckResult>): number {
  const values = Object.values(checks)
  if (values.length === 0) return 0
  return values.reduce((sum, c) => sum + (c.score ?? (c.passed ? 1 : 0)), 0) / values.length
}

function addCheck(checks: Record<string, CheckResult>, id: string, result: CheckResult, flag: { value: boolean }) {
  checks[id] = result
  if (!result.passed) flag.value = false
}

async function runExtraction(transcript: string, currentMemory: string, modelSpec: ModelSpec): Promise<{ raw: string; diff: MemoryDiffResult | null; parseError?: string }> {
  try {
    const providerClient = await getEvalProviderClient()
    const auth = await providerClient.auth.getAuth(modelSpec.provider)
    await providerClient.state.setSelection('secondary', modelSpec.provider, modelSpec.model, auth ?? null, { persist: false })
    const result = await Effect.runPromise(
      Effect.gen(function* () {
        const runtime = yield* ModelResolver
        const model = yield* runtime.resolve('secondary')
        return yield* model.invoke(
          ExtractMemoryDiff,
          { transcript, currentMemory },
        )
      }).pipe(Effect.provide(Layer.merge(makeModelResolver().pipe(Layer.provide(providerClient.layer), Layer.provide(makeNoopTracer())), makeNoopTracer()))),
    )
    return { raw: JSON.stringify(result, null, 2), diff: result }
  } catch (error) {
    return { raw: '', diff: null, parseError: error instanceof Error ? error.message : String(error) }
  }
}

function parseExistingMemoryLines(currentMemory: string): string[] {
  return currentMemory
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.startsWith('- '))
    .map((line) => line.slice(2).trim())
    .filter(Boolean)
}

function applyExpectedChecks(
  checks: Record<string, CheckResult>,
  allPassed: { value: boolean },
  diff: MemoryDiffResult,
  currentMemory: string,
  expected?: MemorySingleScenario['expected']
) {
  if (!expected) return
  if (expected.expectEmpty) addCheck(checks, 'exact-empty', exactEmpty(diff), allPassed)
  addCheck(checks, 'operation-bounds', operationBounds(diff, expected.minTotalOps, expected.maxTotalOps), allPassed)
  if (expected.requiredAdditionCategories?.length) {
    addCheck(checks, 'required-categories', requiredCategories(diff, expected.requiredAdditionCategories), allPassed)
  }
  if (expected.allowedAdditionCategories?.length) {
    addCheck(checks, 'allowed-categories', allowedCategories(diff, expected.allowedAdditionCategories), allPassed)
  }
  if (expected.forbidDuplicateOfExisting) {
    addCheck(checks, 'duplicate-detection', duplicateDetection(diff, currentMemory), allPassed)
  }
  if (expected.expectUpdateOrDeletion) {
    addCheck(checks, 'has-update-or-deletion', hasUpdateOrDeletion(diff), allPassed)
  }
}

async function executeSingleScenario(scenario: MemorySingleScenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  const checks: Record<string, CheckResult> = {}
  const allPassed = { value: true }
  let rawResponse = ''

  const extraction = await runExtraction(scenario.transcript, scenario.currentMemory, modelSpec)
  rawResponse = extraction.raw

  addCheck(checks, 'valid-json-object', validJsonObject(extraction.diff, extraction.parseError), allPassed)
  if (!extraction.diff) {
    return { scenarioId: scenario.id, checks, passed: false, score: computeScore(checks), rawResponse }
  }

  addCheck(checks, 'valid-categories', validCategories(extraction.diff), allPassed)
  addCheck(checks, 'imperative-line-shape', imperativeLineShape(extraction.diff), allPassed)
  applyExpectedChecks(checks, allPassed, extraction.diff, scenario.currentMemory, scenario.expected)

  if (scenario.group === 'quality') {
    const judgePayload = `Transcript:
${scenario.transcript}

Current memory:
${scenario.currentMemory}

Extraction diff:
${rawResponse}`
    const judgeResults = await runJudgeChecks(scenario.judgeChecks, judgePayload)
    for (const [id, jr] of Object.entries(judgeResults)) {
      addCheck(checks, id, { passed: jr.passed, message: jr.message }, allPassed)
    }
  }

  return { scenarioId: scenario.id, checks, passed: allPassed.value, score: computeScore(checks), rawResponse }
}

async function executeMultiScenario(scenario: MemoryMultiScenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  const checks: Record<string, CheckResult> = {}
  const allPassed = { value: true }
  const rawParts: string[] = []
  let memory = scenario.sessions[0]?.currentMemory ?? ''

  for (let i = 0; i < scenario.sessions.length; i++) {
    const session = scenario.sessions[i]
    if (i === 0) {
      memory = session.currentMemory
    } else if (session.currentMemory && session.currentMemory.trim().length > 0) {
      memory = session.currentMemory
    }

    const extraction = await runExtraction(session.transcript, memory, modelSpec)
    rawParts.push(`SESSION_${i + 1}_DIFF:\n${extraction.raw}`)

    addCheck(checks, `session${i + 1}-valid-json-object`, validJsonObject(extraction.diff, extraction.parseError), allPassed)
    if (!extraction.diff) {
      return { scenarioId: scenario.id, checks, passed: false, score: computeScore(checks), rawResponse: rawParts.join('\n\n') }
    }

    addCheck(checks, `session${i + 1}-valid-categories`, validCategories(extraction.diff), allPassed)
    addCheck(checks, `session${i + 1}-imperative-line-shape`, imperativeLineShape(extraction.diff), allPassed)
    applyExpectedChecks(checks, allPassed, extraction.diff, memory, session.expected)

    memory = applyMemoryDiff(memory, extraction.diff).updated
    const existingAfter = parseExistingMemoryLines(memory)
    if (session.expected?.forbidDuplicateOfExisting && existingAfter.length > 0) {
      addCheck(checks, `session${i + 1}-duplicate-detection`, duplicateDetection(extraction.diff, memory), allPassed)
    }
  }

  return { scenarioId: scenario.id, checks, passed: allPassed.value, score: computeScore(checks), rawResponse: rawParts.join('\n\n') }
}

async function executeScenario(scenario: MemoryEvalScenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  if (scenario.group === 'multi') return executeMultiScenario(scenario, modelSpec)
  return executeSingleScenario(scenario, modelSpec)
}

export const memoryExtractionEval: RunnableEval = {
  id: 'memory-extraction',
  name: 'Memory Extraction Eval',
  description: 'Evaluates memory extraction decision quality, multi-session behavior, and judged extraction quality',
  scenarios: ALL_SCENARIOS,
  variants: VARIANTS,
  defaultConcurrency: 3,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario as MemoryEvalScenario, modelSpec)
  },
}