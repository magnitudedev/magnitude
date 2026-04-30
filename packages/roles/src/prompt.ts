import agentCommonRaw from './prompts/shared/agent-common.txt' with { type: 'text' }
import workerBaseRaw from './prompts/shared/worker-base.txt' with { type: 'text' }

/**
 * A strongly typed prompt template. The type parameter captures which
 * runtime variables must be provided at render time.
 *
 * Shared sections (AGENT_COMMON, WORKER_BASE) are resolved at definition
 * time and do not appear as runtime variables.
 */
export type PromptTemplate<TVars extends string = never> = {
  readonly raw: string
  readonly compiled: string
  readonly render: [TVars] extends [never]
    ? () => string
    : (vars: Record<TVars, string>) => string
}

/**
 * Define a prompt template from raw text.
 *
 * Shared sections ({{AGENT_COMMON}}, {{WORKER_BASE}}) are resolved immediately.
 * Any remaining {{PLACEHOLDER}} patterns become typed runtime variables.
 */
export function definePrompt<TVars extends string = never>(
  raw: string,
): PromptTemplate<TVars> {
  const compiled = raw
    .replaceAll('{{AGENT_COMMON}}', agentCommonRaw)
    .replaceAll('{{WORKER_BASE}}', workerBaseRaw)

  const render = ((vars?: Record<string, string>) => {
    if (!vars) return compiled
    let result = compiled
    for (const [key, value] of Object.entries(vars)) {
      result = result.replaceAll(`{{${key}}}`, value)
    }
    return result
  }) as PromptTemplate<TVars>['render']

  return { raw, compiled, render }
}
