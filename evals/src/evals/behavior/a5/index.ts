import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, CheckResult, EvalVariant } from '../../types'
import { callModel } from '../../runner'
import { readFileSync } from 'fs'
import { join } from 'path'
import { getXmlActProtocol } from '@magnitudedev/agent-definition'
import { getAgentDefinition, generateXmlActToolDocs } from '@magnitudedev/agent'
import { ALL_SCENARIOS, type A5Scenario } from './scenarios'

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
      .replaceAll('{{RESPONSE_PROTOCOL}}', getXmlActProtocol('user', agentDef.thinkingLenses.slice()))
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

async function runJudge(
  question: string,
  orchestratorResponse: string
): Promise<boolean> {
  const prompt = `Here is an AI assistant's response:\n\n<response>\n${orchestratorResponse}\n</response>\n\nQuestion: ${question}\n\nAnswer with only "yes" or "no".`
  const result = await callModel(JUDGE_SYSTEM_PROMPT, [{ role: 'user', content: [prompt] }], JUDGE_MODEL_SPEC)
  return result.trim().toLowerCase().startsWith('yes')
}

async function executeScenario(scenario: A5Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  let rawResponse = ''
  try {
    rawResponse = await callModel(getSystemPrompt(), scenario.messages, modelSpec)
  } catch (error) {
    const checks: Record<string, CheckResult> = {}
    const allChecks = [...scenario.checks, ...(scenario.judgeChecks ?? [])]
    for (const check of allChecks) {
      checks[check.id] = {
        passed: false,
        message: `Model call failed: ${error instanceof Error ? error.message : String(error)}`,
      }
    }
    return { scenarioId: scenario.id, checks, passed: false, score: 0, rawResponse: '' }
  }

  const checks: Record<string, CheckResult> = {}
  let allPassed = true

  // Structural checks
  for (const check of scenario.checks) {
    const result = check.evaluate(rawResponse, null as never)
    checks[check.id] = result
    if (!result.passed) allPassed = false
  }

  // Judge checks
  for (const judgeCheck of scenario.judgeChecks ?? []) {
    try {
      const passed = await runJudge(judgeCheck.question, rawResponse)
      checks[judgeCheck.id] = {
        passed,
        message: passed ? undefined : `Judge answered "no" to: ${judgeCheck.question}`,
      }
      if (!passed) allPassed = false
    } catch (error) {
      checks[judgeCheck.id] = {
        passed: false,
        message: `Judge call failed: ${error instanceof Error ? error.message : String(error)}`,
      }
      allPassed = false
    }
  }

  const avgScore = Object.values(checks).length > 0
    ? Object.values(checks).reduce((sum, c) => sum + (c.score ?? (c.passed ? 1 : 0)), 0) / Object.values(checks).length
    : 0

  return { scenarioId: scenario.id, checks, passed: allPassed, score: avgScore, rawResponse }
}

const VARIANTS: EvalVariant[] = [
  { id: 'tenet1', label: 'Tenet 1 — Communicate Before Acting', count: 2 },
  { id: 'tenet2', label: 'Tenet 2 — Surface Assumptions', count: 2 },
  { id: 'tenet3', label: 'Tenet 3 — Resolve Autonomously', count: 2 },
]

export const a5Eval: RunnableEval = {
  id: 'a5',
  name: 'A5 Behavioral Eval',
  description: 'Tests orchestrator adherence to the A5 behavioral tenets',
  scenarios: ALL_SCENARIOS,
  variants: VARIANTS,
  defaultConcurrency: 3,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario as A5Scenario, modelSpec)
  },
}