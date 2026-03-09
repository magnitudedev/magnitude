/**
 * Core types for the evals framework
 */

import type { TestSandboxResult } from './test-sandbox'
import type { ChatMessage } from '@magnitudedev/llm-core'
export type { ChatMessage } from '@magnitudedev/llm-core'


/**
 * A single check within a scenario that verifies a specific property of the LLM response
 */
export interface Check {
  /** Unique identifier for this check */
  id: string
  /** Human-readable description of what this check verifies */
  description: string
  /** 
   * Evaluate the LLM response against this check.
   * @param rawResponse - The raw LLM output text
   * @param result - Full test sandbox result with captured tool calls and events
   */
  evaluate(rawResponse: string, result: TestSandboxResult): CheckResult
}

/**
 * Result of a single check evaluation
 */
export interface CheckResult {
  passed: boolean
  /** 0-1, defaults to 1 if passed, 0 if not */
  score?: number
  /** Explanation of why the check failed */
  message?: string
  /** Relevant snippet from the response showing the issue */
  snippet?: string
}

/**
 * A test scenario -- a synthetic conversation that elicits specific LLM behavior
 */
export interface Scenario {
  /** Unique identifier */
  id: string
  /** What this scenario tests */
  description: string
  /** The conversation history to send to the LLM */
  messages: ChatMessage[]
  /** Checks to run against the response */
  checks: Check[]
}

/**
 * A named variant group within an eval.
 * Scenario IDs must be prefixed with `{variant.id}/`.
 */
export interface EvalVariant {
  /** Identifier — matches the scenario ID prefix before the first `/` */
  id: string
  /** Human-readable label for interactive selection */
  label: string
  /** Number of scenarios in this variant */
  count: number
}

/**
 * An eval -- a collection of related scenarios testing a specific capability
 */
export interface Eval {
  /** Unique identifier */
  id: string
  /** Human-readable name */
  name: string
  /** What this eval tests */
  description: string
  /** The scenarios in this eval */
  scenarios: Scenario[]
  /** Named variant groups — when present, CLI offers interactive variant selection */
  variants?: EvalVariant[]
  /** Default concurrency for this eval (overridden by CLI --concurrency) */
  defaultConcurrency?: number
  /** Hardcoded model list — when present, --all-models or no -m flags uses these */
  defaultModels?: ModelSpec[]
}

/**
 * An eval that provides its own scenario execution pipeline,
 * bypassing the default sandbox-based runner.
 */
export interface RunnableEval extends Eval {
  runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult>
}

export function isRunnableEval(eval_: Eval): eval_ is RunnableEval {
  return 'runScenario' in eval_
}

/**
 * Result of evaluating a single scenario
 */
export interface ScenarioResult {
  scenarioId: string
  checks: Record<string, CheckResult>
  passed: boolean
  /** Average of check scores */
  score: number
  rawResponse: string
  /** Which run this is when using --repeat (0-indexed) */
  runIndex?: number
}

/**
 * Result of running an eval against a single model
 */
export interface EvalRunResult {
  evalId: string
  model: string
  provider: string
  scenarios: ScenarioResult[]
  passedCount: number
  totalCount: number
  averageScore: number
}

/**
 * Model specification as provider:model pair
 */
export interface ModelSpec {
  provider: string
  model: string
  label: string
}

/**
 * Parse a "provider:model" string into a ModelSpec
 */
export function parseModelSpec(spec: string): ModelSpec {
  const colonIdx = spec.indexOf(':')
  if (colonIdx === -1) {
    throw new Error(`Invalid model spec "${spec}" -- expected "provider:model" format`)
  }
  const provider = spec.slice(0, colonIdx)
  const model = spec.slice(colonIdx + 1)
  return { provider, model, label: spec }
}
