import type { RunnableEval, Scenario, ScenarioResult, ModelSpec, CheckResult, EvalVariant } from '../../../types'
import { callModel } from '../../../runner'
import { readFileSync } from 'fs'
import { join } from 'path'
import { getXmlActProtocol } from '@magnitudedev/agent'
import { getAgentDefinition, generateXmlActToolDocs } from '@magnitudedev/agent'
import { ALL_SCENARIOS, type ToolUsageScenario } from './scenarios'

let cachedSystemPrompt: string | null = null

function getSystemPrompt(): string {
  if (!cachedSystemPrompt) {
    const raw = readFileSync(
      join(__dirname, '../../../../../packages/agent/src/agents/prompts/orchestrator.txt'),
      'utf-8'
    )
    const agentDef = getAgentDefinition('orchestrator')
    const toolDocs = generateXmlActToolDocs(agentDef, [])
    cachedSystemPrompt = raw
      .replaceAll('{{RESPONSE_PROTOCOL}}', getXmlActProtocol('user', agentDef.lenses.slice()))
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

async function runJudge(question: string, orchestratorResponse: string): Promise<boolean> {
  const prompt = `Here is an AI assistant's response:\n\n<response>\n${orchestratorResponse}\n</response>\n\nQuestion: ${question}\n\nAnswer with only "yes" or "no".`
  const result = await callModel(JUDGE_SYSTEM_PROMPT, [{ role: 'user', content: [prompt] }], JUDGE_MODEL_SPEC)
  return result.trim().toLowerCase().startsWith('yes')
}

async function executeScenario(scenario: ToolUsageScenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
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

  for (const check of scenario.checks) {
    const result = check.evaluate(rawResponse, null as never)
    checks[check.id] = result
    if (!result.passed) allPassed = false
  }

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

export const VARIANTS: EvalVariant[] = [
  { id: 'tool-usage/explorer-vs-reads', label: 'Explorer vs direct reads', count: 1 },
  { id: 'tool-usage/builder-vs-edits', label: 'Builder vs direct edits', count: 1 },
  { id: 'tool-usage/planner-vs-self-plan', label: 'Planner vs self-planning', count: 1 },
  { id: 'tool-usage/explorer-vs-direct-reads-depth', label: 'Explorer depth vs direct reads', count: 1 },
]

export const toolUsageScenarios: Scenario[] = ALL_SCENARIOS

export async function runToolUsageScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  return executeScenario(scenario as ToolUsageScenario, modelSpec)
}