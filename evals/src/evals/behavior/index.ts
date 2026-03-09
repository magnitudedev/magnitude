import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, EvalVariant } from '../../types'
import { ALL_SCENARIOS as A5_SCENARIOS_RAW, type A5Scenario } from './a5/scenarios'
import { toolUsageScenarios, runToolUsageScenario } from './tool-usage/index'

// Prefix a5 scenario IDs so they group under the 'a5' variant
const A5_SCENARIOS: A5Scenario[] = A5_SCENARIOS_RAW.map(s => ({ ...s, id: `a5/${s.id}` }))

// Re-use a5's execution logic
import { readFileSync } from 'fs'
import { join } from 'path'
import { getXmlActProtocol, builtInThinkingLenses } from '@magnitudedev/agent-definition'
import { getAgentDefinition, generateXmlActToolDocs } from '@magnitudedev/agent'
import { callModel } from '../../runner'

let cachedSystemPrompt: string | null = null

function getSystemPrompt(): string {
  if (!cachedSystemPrompt) {
    const raw = readFileSync(
      join(__dirname, '../../../../packages/agent/src/agents/prompts/orchestrator.txt'),
      'utf-8'
    )
    const agentDef = getAgentDefinition('orchestrator')
    const toolDocs = generateXmlActToolDocs(agentDef, ['think'])
    cachedSystemPrompt = raw
      .replaceAll('{{RESPONSE_PROTOCOL}}', getXmlActProtocol('user', builtInThinkingLenses.slice()))
      .replaceAll('{{TOOL_DOCS}}', toolDocs)
  }
  return cachedSystemPrompt
}

const JUDGE_SYSTEM_PROMPT = `You are an objective evaluator. You will be shown a conversation and a response, then asked a yes/no question about the response. Answer with only "yes" or "no".`

const JUDGE_MODEL_SPEC: ModelSpec = {
  provider: 'anthropic',
  model: 'claude-sonnet-4-6',
  label: 'anthropic:claude-sonnet-4-6',
}

async function runJudge(question: string, response: string): Promise<boolean> {
  const prompt = `Here is an AI assistant's response:\n\n<response>\n${response}\n</response>\n\nQuestion: ${question}\n\nAnswer with only "yes" or "no".`
  const result = await callModel(JUDGE_SYSTEM_PROMPT, [{ role: 'user', content: [prompt] }], JUDGE_MODEL_SPEC)
  return result.trim().toLowerCase().startsWith('yes')
}

async function runA5Scenario(scenario: A5Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  let rawResponse = ''
  try {
    rawResponse = await callModel(getSystemPrompt(), scenario.messages, modelSpec)
  } catch (error) {
    const allChecks = [...scenario.checks, ...(scenario.judgeChecks ?? [])]
    const checks: Record<string, import('../../types').CheckResult> = {}
    for (const check of allChecks) {
      checks[check.id] = { passed: false, message: `Model call failed: ${error instanceof Error ? error.message : String(error)}` }
    }
    return { scenarioId: scenario.id, checks, passed: false, score: 0, rawResponse: '' }
  }

  const checks: Record<string, import('../../types').CheckResult> = {}
  let allPassed = true

  for (const check of scenario.checks) {
    const result = check.evaluate(rawResponse, null as never)
    checks[check.id] = result
    if (!result.passed) allPassed = false
  }

  for (const judgeCheck of scenario.judgeChecks ?? []) {
    try {
      const passed = await runJudge(judgeCheck.question, rawResponse)
      checks[judgeCheck.id] = { passed, message: passed ? undefined : `Judge answered "no" to: ${judgeCheck.question}` }
      if (!passed) allPassed = false
    } catch (error) {
      checks[judgeCheck.id] = { passed: false, message: `Judge call failed: ${error instanceof Error ? error.message : String(error)}` }
      allPassed = false
    }
  }

  const avgScore = Object.values(checks).length > 0
    ? Object.values(checks).reduce((sum, c) => sum + (c.score ?? (c.passed ? 1 : 0)), 0) / Object.values(checks).length
    : 0

  return { scenarioId: scenario.id, checks, passed: allPassed, score: avgScore, rawResponse }
}

const ALL_SCENARIOS: Scenario[] = [
  ...A5_SCENARIOS,
  ...toolUsageScenarios,
]

const VARIANTS: EvalVariant[] = [
  { id: 'a5', label: 'A5 Behavioral Tenets', count: 6 },
  { id: 'tool-usage', label: 'Tool & Subagent Usage', count: 4 },
]

export const behaviorEval: RunnableEval = {
  id: 'behavior',
  name: 'Orchestrator Behavior Eval',
  description: 'Tests orchestrator behavioral correctness across A5 tenets and tool/subagent usage',
  scenarios: ALL_SCENARIOS,
  variants: VARIANTS,
  defaultConcurrency: 3,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    if (scenario.id.startsWith('tool-usage/')) {
      return runToolUsageScenario(scenario, modelSpec)
    }
    // Strip the 'a5/' prefix before passing to a5 runner
    const a5Scenario = { ...scenario, id: scenario.id.replace(/^a5\//, '') } as A5Scenario
    return runA5Scenario(a5Scenario, modelSpec)
  },
}