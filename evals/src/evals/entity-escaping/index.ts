import type { RunnableEval, EvalVariant, Scenario, ScenarioResult, ModelSpec, Check, CheckResult } from '../../types'
import { callModel } from '../../runner'

const SYSTEM_PROMPTS = {
  'no-instruction': `You respond using XML tool calls.

You have access to one tool:

"write" - writes content to a file. It is an XML element with tag name "write", a "path" attribute, and the file content as the element body.

Example:
<write path="hello.txt">Hello world</write>

Respond with only the tool call XML. Do not include any other text.`,
  'anti-escape': `You respond using XML tool calls.

You have access to one tool:

"write" - writes content to a file. It is an XML element with tag name "write", a "path" attribute, and the file content as the element body.

Example:
<write path="hello.txt">Hello world</write>

IMPORTANT: This is NOT a standard XML parser. It reads tag structure but treats all content as raw text — no entity processing, no escaping.
WRONG: \`&lt;div&gt;\`, \`&amp;\`, \`&quot;\`
RIGHT: \`<div>\`, \`&\`, \`"\`
This applies everywhere — action bodies and action attributes. Escaping will corrupt your output.

Respond with only the tool call XML. Do not include any other text.`,
} as const

type VariantId = keyof typeof SYSTEM_PROMPTS

interface EntityEscapingScenarioDef {
  id: string
  description: string
  userMessage: string
}

const SCENARIO_DEFS: EntityEscapingScenarioDef[] = [
  {
    id: 'react-component',
    description: 'React functional component with JSX tags',
    userMessage: `Write a React functional component in TypeScript at \`App.tsx\` that renders a div with an h1 saying 'Hello' and a paragraph saying 'World'.`,
  },
  {
    id: 'html-template',
    description: 'Basic HTML page template',
    userMessage: `Write an HTML file at \`index.html\` with a basic page structure: doctype, html, head with title 'Test', and body with a div containing a heading and paragraph.`,
  },
  {
    id: 'jsx-conditional',
    description: 'React component with conditional JSX rendering',
    userMessage: `Write a React component at \`List.tsx\` that takes a \`items: string[]\` prop and renders: if items.length > 0, a ul with li elements for each item; otherwise a p saying 'No items'.`,
  },
  {
    id: 'jsx-event-handlers',
    description: 'React component with event handlers and state',
    userMessage: `Write a React component at \`Button.tsx\` with a button that has onClick, onMouseEnter, and onMouseLeave handlers, using useState for a hover state.`,
  },
  {
    id: 'html-form',
    description: 'HTML form with labeled inputs',
    userMessage: `Write an HTML file at \`form.html\` with a form containing labeled input fields for name, email, and a textarea for message, plus a submit button.`,
  },
  {
    id: 'tsx-generic',
    description: 'Generic TSX component with type constraints',
    userMessage: `Write a TypeScript React component at \`Select.tsx\` that is generic: \`function Select<T extends { id: string; label: string }>(props: { options: T[]; onSelect: (item: T) => void })\` — render a select element with options.`,
  },
  {
    id: 'ts-generics',
    description: 'TypeScript generic types and utility functions',
    userMessage: `Write a TypeScript file at \`types.ts\` that exports: a generic type \`Result<T, E>\` (a discriminated union with \`{ ok: true; value: T }\` and \`{ ok: false; error: E }\`), a function \`mapResult<T, U, E>(result: Result<T, E>, fn: (value: T) => U): Result<U, E>\`, and a type \`Registry<K extends string, V>\` that wraps \`Map<K, Array<V>>\`.`,
  },
  {
    id: 'svg-component',
    description: 'React SVG component',
    userMessage: `Write a React component at \`Icon.tsx\` that renders an SVG icon: a 24x24 viewBox with a circle and a path element.`,
  },
]

const DEFAULT_MODELS: ModelSpec[] = [
  { provider: 'openai', model: 'gpt-5.3-codex', label: 'openai:gpt-5.3-codex' },
  { provider: 'anthropic', model: 'claude-sonnet-4-6', label: 'anthropic:claude-sonnet-4-6' },
]

interface EntityAnalysis {
  hasWriteTag: boolean
  body: string | null
  hasEntityLt: boolean
  hasEntityGt: boolean
  hasEntityAmp: boolean
  hasEntityQuot: boolean
  hasAnyEntity: boolean
  entityCount: number
  hasRawAngleBrackets: boolean
  escapedLines: string[]
}

function makeScenario(variant: VariantId, def: EntityEscapingScenarioDef): Scenario {
  return {
    id: `${variant}/${def.id}`,
    description: `[${variant}] ${def.description}`,
    messages: [{ role: 'user', content: [def.userMessage] }],
    checks: buildChecks(),
  }
}

function buildChecks(): Check[] {
  return [
    {
      id: 'has-any-entity',
      description: 'Primary metric: fails if body contains any target entity escapes',
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse)
        const passed = analysis.hasWriteTag && !analysis.hasAnyEntity
        return {
          passed,
          score: passed ? 1 : 0,
          message: analysis.hasWriteTag
            ? (analysis.hasAnyEntity
              ? `found-entities:${analysis.entityCount}\n${analysis.escapedLines.slice(0, 5).join('\n')}`
              : 'no-entities')
            : 'missing-write-tag',
          snippet: analysis.body ?? undefined,
        }
      },
    },
    {
      id: 'entity-count',
      description: 'Scores lower as number of entity escapes increases',
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse)
        if (!analysis.hasWriteTag) {
          return {
            passed: false,
            score: 0,
            message: 'missing-write-tag',
            snippet: undefined,
          }
        }

        const score = analysis.entityCount === 0 ? 1 : 1 / (1 + analysis.entityCount)
        return {
          passed: analysis.entityCount === 0,
          score,
          message: `entity-count=${analysis.entityCount}`,
          snippet: analysis.body ?? undefined,
        }
      },
    },
    {
      id: 'has-raw-angle-brackets',
      description: 'Checks whether body includes raw tag-like syntax (< followed by alpha)',
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse)
        const passed = analysis.hasWriteTag && analysis.hasRawAngleBrackets
        const bodyPreview = analysis.body
          ? analysis.body.split('\n').slice(0, 5).map((l, i) => `L${i+1}: ${l.trim()}`).join('\n')
          : ''
        return {
          passed,
          score: passed ? 1 : 0,
          message: analysis.hasWriteTag
            ? (analysis.hasRawAngleBrackets
              ? 'has-raw-angle-brackets'
              : `no-raw-angle-brackets\n${bodyPreview}`)
            : 'missing-write-tag',
          snippet: analysis.body ?? undefined,
        }
      },
    },
  ]
}

function analyzeResponse(rawResponse: string): EntityAnalysis {
  const writeMatch = rawResponse.match(/<write\b[^>]*>([\s\S]*?)<\/write>/i)
  if (!writeMatch) {
    return {
      hasWriteTag: false,
      body: null,
      hasEntityLt: false,
      hasEntityGt: false,
      hasEntityAmp: false,
      hasEntityQuot: false,
      hasAnyEntity: false,
      entityCount: 0,
      hasRawAngleBrackets: false,
      escapedLines: [],
    }
  }

  const body = writeMatch[1] ?? ''
  const ltCount = countMatches(body, /&lt;/g)
  const gtCount = countMatches(body, /&gt;/g)
  const ampCount = countMatches(body, /&amp;/g)
  const quotCount = countMatches(body, /&quot;/g)
  const entityCount = ltCount + gtCount + ampCount + quotCount
  const escapedLines = body
    .split('\n')
    .map((line, i) => ({ line, lineNum: i + 1 }))
    .filter(({ line }) => /&lt;|&gt;|&amp;|&quot;/.test(line))
    .map(({ line, lineNum }) => `L${lineNum}: ${line.trim()}`)

  return {
    hasWriteTag: true,
    body,
    hasEntityLt: ltCount > 0,
    hasEntityGt: gtCount > 0,
    hasEntityAmp: ampCount > 0,
    hasEntityQuot: quotCount > 0,
    hasAnyEntity: entityCount > 0,
    entityCount,
    hasRawAngleBrackets: /<[A-Za-z]/.test(body),
    escapedLines,
  }
}

function countMatches(input: string, regex: RegExp): number {
  return input.match(regex)?.length ?? 0
}

function parseScenarioId(id: string): { variant: VariantId; baseId: string } {
  const match = id.match(/^(no-instruction|anti-escape)\/(.+)$/)
  if (!match) throw new Error(`Invalid scenario ID: ${id}`)
  return { variant: match[1] as VariantId, baseId: match[2] }
}

function makeFail(scenario: Scenario, message: string, rawResponse = ''): ScenarioResult {
  const checks = Object.fromEntries(
    scenario.checks.map((check) => [check.id, { passed: false, score: 0, message } satisfies CheckResult]),
  )
  return {
    scenarioId: scenario.id,
    checks,
    passed: false,
    score: 0,
    rawResponse,
  }
}

async function executeScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
  const { variant } = parseScenarioId(scenario.id)

  let rawResponse: string
  try {
    rawResponse = await callModel(SYSTEM_PROMPTS[variant], scenario.messages, modelSpec)
  } catch (error) {
    return makeFail(scenario, `Model call failed: ${error instanceof Error ? error.message : String(error)}`)
  }

  const checks: Record<string, CheckResult> = {}
  let passed = true
  for (const check of scenario.checks) {
    const result = check.evaluate(rawResponse, null as never)
    checks[check.id] = result
    if (!result.passed) passed = false
  }

  const scoreValues = Object.values(checks)
  const score = scoreValues.length > 0
    ? scoreValues.reduce((sum, result) => sum + (result.score ?? (result.passed ? 1 : 0)), 0) / scoreValues.length
    : 0

  return {
    scenarioId: scenario.id,
    checks,
    passed,
    score,
    rawResponse,
  }
}

const variants: EvalVariant[] = (Object.keys(SYSTEM_PROMPTS) as VariantId[]).map((variant) => ({
  id: variant,
  label: variant,
  count: SCENARIO_DEFS.length,
}))

const scenarios = (Object.keys(SYSTEM_PROMPTS) as VariantId[]).flatMap((variant) =>
  SCENARIO_DEFS.map((def) => makeScenario(variant, def)),
)

export const entityEscapingEval: RunnableEval = {
  id: 'entity-escaping',
  name: 'Entity Escaping Bias',
  description: `Measures entity-escaping bias in XML write tool bodies across ${variants.length} prompt variants and ${SCENARIO_DEFS.length} scenarios (${scenarios.length} total runs per model)`,
  scenarios,
  variants,
  defaultConcurrency: 4,
  defaultModels: DEFAULT_MODELS,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario, modelSpec)
  },
}
