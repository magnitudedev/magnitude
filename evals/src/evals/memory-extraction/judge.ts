import { callModel } from '../../runner'
import type { ModelSpec } from '../../types'
import type { JudgeCheck } from './types'

const JUDGE_SYSTEM_PROMPT = `You are an objective evaluator for memory extraction quality.
Prioritize: durability across sessions, precision/minimality, and correct category fit.
Answer only "yes" or "no".`

const JUDGE_MODEL_SPEC: ModelSpec = {
  provider: 'anthropic',
  model: 'claude-sonnet-4-6',
  label: 'anthropic:claude-sonnet-4-6',
}

export async function runJudge(question: string, payload: string): Promise<boolean> {
  const prompt = `Evaluate this extraction result.

<context>
${payload}
</context>

Question: ${question}

Answer with only "yes" or "no".`
  const result = await callModel(JUDGE_SYSTEM_PROMPT, [{ role: 'user', content: [prompt] }], JUDGE_MODEL_SPEC)
  return result.trim().toLowerCase().startsWith('yes')
}

export async function runJudgeChecks(
  checks: JudgeCheck[] | undefined,
  payload: string
): Promise<Record<string, { passed: boolean; message?: string }>> {
  const out: Record<string, { passed: boolean; message?: string }> = {}
  for (const check of checks ?? []) {
    try {
      const passed = await runJudge(check.question, payload)
      out[check.id] = { passed, message: passed ? undefined : `Judge answered "no" to: ${check.question}` }
    } catch (error) {
      out[check.id] = { passed: false, message: `Judge call failed: ${error instanceof Error ? error.message : String(error)}` }
    }
  }
  return out
}