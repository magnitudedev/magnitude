import type { RunnableEval, EvalVariant, Scenario, ScenarioResult, ModelSpec, Check, CheckResult } from '../../types'
import { callModel } from '../../runner'

const SYSTEM_PROMPTS = {
  neutral: `You have access to two XML tools.

Tool 1: "write" - writes content to a file. It is an XML element with tag name "write", a "path" attribute, and the file content as the element body.

Tool 2: "edit" - edits a file by replacing text. It is an XML element with tag name "edit" and a "path" attribute. It has two child elements: "old" containing the exact text to find, and "new" containing the replacement text.

Respond with only the tool call XML. Do not include any other text.`,
  inline: `You have access to two XML tools.

Tool 1: "write" - writes content to a file. It is an XML element with tag name "write", a "path" attribute, and the file content as the element body.

Tool 2: "edit" - edits a file by replacing text. It is an XML element with tag name "edit" and a "path" attribute. It has two child elements: "old" containing the exact text to find, and "new" containing the replacement text.

IMPORTANT: Content between tags is interpreted literally. Start content immediately after the opening tag's > character. Place the closing tag immediately after the content ends. Every character between the opening and closing tags, including newlines, is part of the content.

Respond with only the tool call XML. Do not include any other text.`,
  block: `You have access to two XML tools.

Tool 1: "write" - writes content to a file. It is an XML element with tag name "write", a "path" attribute, and the file content as the element body.

Tool 2: "edit" - edits a file by replacing text. It is an XML element with tag name "edit" and a "path" attribute. It has two child elements: "old" containing the exact text to find, and "new" containing the replacement text.

IMPORTANT: Place content on the line after the opening tag, and place the closing tag on its own line after the content. One leading newline (after opening tag) and one trailing newline (before closing tag) are automatically stripped and are not part of the content.

Respond with only the tool call XML. Do not include any other text.`,
} as const

type VariantId = keyof typeof SYSTEM_PROMPTS
type ScenarioGroup = 'write' | 'edit'

interface XmlNewlineScenarioDef {
  id: string
  description: string
  group: ScenarioGroup
  userMessage: string
  tags: string[]
  trailingNewlineFocus?: boolean
  expectedPrimaryBody?: string
  expectedTrailingNewline?: boolean
}

const SCENARIO_DEFS: XmlNewlineScenarioDef[] = [
  {
    id: 'a1-write-single-line',
    description: 'Write tool with single-line body content',
    group: 'write',
    tags: ['write'],
    userMessage: 'Write a file `greeting.txt` containing: hello world',
    expectedPrimaryBody: 'hello world',
  },
  {
    id: 'a2-write-two-lines',
    description: 'Write tool with two-line body content',
    group: 'write',
    tags: ['write'],
    userMessage: `Write a file \`config.txt\` containing these two lines:
host=localhost
port=3000`,
    expectedPrimaryBody: 'host=localhost\nport=3000',
  },
  {
    id: 'a3-write-code-function',
    description: 'Write tool with multiline code body content',
    group: 'write',
    tags: ['write'],
    userMessage: `Write a file \`add.js\` containing:
function add(a, b) {
  return a + b;
}`,
    expectedPrimaryBody: 'function add(a, b) {\n  return a + b;\n}',
  },
  {
    id: 'a4-write-indented-content',
    description: 'Write tool with indented structured content',
    group: 'write',
    tags: ['write'],
    userMessage: `Write a file \`data.yaml\` containing:
items:
  - name: foo
    value: 1
  - name: bar
    value: 2`,
    expectedPrimaryBody: 'items:\n  - name: foo\n    value: 1\n  - name: bar\n    value: 2',
  },
  {
    id: 'a5-write-single-char',
    description: 'Write tool with a single character body',
    group: 'write',
    tags: ['write'],
    userMessage: 'Write a file `x.txt` containing just the letter: x',
    expectedPrimaryBody: 'x',
  },
  {
    id: 'a6-write-empty-lines-in-content',
    description: 'Write tool with an internal blank line',
    group: 'write',
    tags: ['write'],
    userMessage: `Write a file \`spaced.txt\` containing:
line1

line3`,
    expectedPrimaryBody: 'line1\n\nline3',
  },
  {
    id: 'b1-write-trailing-newline-explicit',
    description: 'Write tool with explicit trailing newline request',
    group: 'write',
    tags: ['write'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: true,
    expectedPrimaryBody: 'hello\n',
    userMessage: 'Write a file `msg.txt` containing the text "hello" followed by a newline character. The file should be exactly 6 bytes: h, e, l, l, o, \\n',
  },
  {
    id: 'b2-write-no-trailing-newline-explicit',
    description: 'Write tool with explicit no-trailing-newline request',
    group: 'write',
    tags: ['write'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: false,
    expectedPrimaryBody: 'hello',
    userMessage: 'Write a file `msg.txt` containing exactly the text "hello" with no trailing newline. The file should be exactly 5 bytes: h, e, l, l, o',
  },
  {
    id: 'b3-write-multiline-with-trailing-newline',
    description: 'Write tool with multiline content ending in a newline',
    group: 'write',
    tags: ['write'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: true,
    expectedPrimaryBody: 'alpha\nbeta\n',
    userMessage: 'Write a file `data.txt` containing two lines "alpha" and "beta", each followed by a newline. The file should end with a newline character.',
  },
  {
    id: 'b4-write-multiline-no-trailing-newline',
    description: 'Write tool with multiline content and no trailing newline',
    group: 'write',
    tags: ['write'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: false,
    expectedPrimaryBody: 'alpha\nbeta',
    userMessage: 'Write a file `data.txt` containing "alpha\\nbeta" with no trailing newline. The file should be exactly 10 bytes.',
  },
  {
    id: 'b5-write-code-with-trailing-newline',
    description: 'Write tool with code ending in a newline',
    group: 'write',
    tags: ['write'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: true,
    expectedPrimaryBody: 'console.log("hi");\n',
    userMessage: 'Write a file `main.js` containing `console.log("hi");` followed by a final newline character.',
  },
  {
    id: 'b6-write-code-no-trailing-newline',
    description: 'Write tool with code and no trailing newline',
    group: 'write',
    tags: ['write'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: false,
    expectedPrimaryBody: 'console.log("hi");',
    userMessage: 'Write a file `main.js` containing exactly `console.log("hi");` with no trailing newline.',
  },
  {
    id: 'c1-edit-single-line',
    description: 'Edit tool with single-line old/new child content',
    group: 'edit',
    tags: ['old', 'new'],
    userMessage: `Given file \`app.js\`:
\`\`\`
const port = 3000;
const host = "localhost";
\`\`\`
Replace \`const port = 3000;\` with \`const port = 8080;\``,
    expectedPrimaryBody: 'const port = 3000;',
  },
  {
    id: 'c2-edit-multiline',
    description: 'Edit tool with multiline child content',
    group: 'edit',
    tags: ['old', 'new'],
    userMessage: `Given file \`app.js\`:
\`\`\`
function greet() {
  console.log("hello");
  return true;
}
\`\`\`
Replace the function body (the two lines inside the braces) with:
  console.log("goodbye");
  return false;`,
    expectedPrimaryBody: '  console.log("hello");\n  return true;',
  },
  {
    id: 'c3-edit-indented-block',
    description: 'Edit tool replacing an indented JSON line',
    group: 'edit',
    tags: ['old', 'new'],
    userMessage: `Given file \`config.json\`:
\`\`\`
{
  "name": "app",
  "version": "1.0.0",
  "main": "index.js"
}
\`\`\`
Replace \`"version": "1.0.0",\` with \`"version": "2.0.0",\``,
    expectedPrimaryBody: '"version": "1.0.0",',
  },
  {
    id: 'c4-edit-single-word',
    description: 'Edit tool replacing a single word',
    group: 'edit',
    tags: ['old', 'new'],
    userMessage: `Given file \`readme.md\`:
\`\`\`
# Hello World
This is a test.
\`\`\`
Replace \`Hello\` with \`Goodbye\``,
    expectedPrimaryBody: 'Hello',
  },
  {
    id: 'c5-edit-entire-line',
    description: 'Edit tool replacing an entire line',
    group: 'edit',
    tags: ['old', 'new'],
    userMessage: `Given file \`list.txt\`:
\`\`\`
apple
banana
cherry
\`\`\`
Replace \`banana\` with \`blueberry\``,
    expectedPrimaryBody: 'banana',
  },
  {
    id: 'd1-edit-add-blank-line',
    description: 'Edit tool inserting a blank line',
    group: 'edit',
    tags: ['old', 'new'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: true,
    expectedPrimaryBody: 'beta\n',
    userMessage: `Given file \`data.txt\`:
\`\`\`
alpha
beta
gamma
\`\`\`
Replace \`beta\` with \`beta\` followed by a newline character (so there's a blank line between beta and gamma)`,
  },
  {
    id: 'd2-edit-remove-blank-line',
    description: 'Edit tool removing a blank line',
    group: 'edit',
    tags: ['old', 'new'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: false,
    expectedPrimaryBody: 'beta\n',
    userMessage: `Given file \`data.txt\`:
\`\`\`
alpha
beta

gamma
\`\`\`
Remove the blank line between beta and gamma`,
  },
  {
    id: 'd3-edit-add-trailing-newline-to-last-line',
    description: 'Edit tool adding a trailing newline to the last line',
    group: 'edit',
    tags: ['old', 'new'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: true,
    expectedPrimaryBody: 'world',
    userMessage: `Given file \`data.txt\` (no trailing newline):
\`\`\`
hello
world
\`\`\`
The file currently has no trailing newline after "world". Add a trailing newline so the file ends with a newline character after "world".`,
  },
  {
    id: 'd4-edit-remove-trailing-newline',
    description: 'Edit tool removing a trailing newline',
    group: 'edit',
    tags: ['old', 'new'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: false,
    expectedPrimaryBody: 'world\n',
    userMessage: `Given file \`data.txt\` (has trailing newline):
\`\`\`
hello
world

\`\`\`
Remove the trailing blank line so the file ends immediately after "world" with no trailing newline.`,
  },
  {
    id: 'd5-edit-multiple-blank-lines',
    description: 'Edit tool reducing two blank lines to one',
    group: 'edit',
    tags: ['old', 'new'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: false,
    expectedPrimaryBody: '\n\n',
    userMessage: `Given file \`data.txt\`:
\`\`\`
section1


section2
\`\`\`
Replace the two blank lines between section1 and section2 with a single blank line.`,
  },
  {
    id: 'd6-edit-preserve-internal-blank-line',
    description: 'Edit tool preserving an internal blank line while editing nearby text',
    group: 'edit',
    tags: ['old', 'new'],
    trailingNewlineFocus: true,
    expectedTrailingNewline: false,
    expectedPrimaryBody: 'body',
    userMessage: `Given file \`data.txt\`:
\`\`\`
header

body
footer
\`\`\`
Replace \`body\` with \`new body\` (preserving the blank line above it)`,
  },
]

const DEFAULT_MODELS: ModelSpec[] = [
  { provider: 'anthropic', model: 'claude-sonnet-4-6', label: 'anthropic:claude-sonnet-4-6' },
  { provider: 'anthropic', model: 'claude-haiku-4-5', label: 'anthropic:claude-haiku-4-5' },
  { provider: 'openai', model: 'gpt-5.3-codex', label: 'openai:gpt-5.3-codex' },
  { provider: 'google', model: 'gemini-3.1-pro-preview', label: 'google:gemini-3.1-pro-preview' },
]

function makeScenario(variant: VariantId, def: XmlNewlineScenarioDef): Scenario {
  return {
    id: `${variant}/${def.id}`,
    description: `[${variant}] ${def.description}`,
    messages: [{ role: 'user', content: [def.userMessage] }],
    checks: buildChecks(variant, def),
  }
}

function buildChecks(variant: VariantId, def: XmlNewlineScenarioDef): Check[] {
  const checks: Check[] = [
    {
      id: 'body-format',
      description: `Classify whether ${def.tags.join('/')} content starts inline or on the next line`,
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse, def)
        const passed = analysis.bodyFormat !== 'other'
        return {
          passed,
          score: passed ? 1 : 0,
          message: analysis.bodyFormat,
          snippet: analysis.primaryBody ?? undefined,
        }
      },
    },
    {
      id: 'closing-format',
      description: `Classify whether ${def.tags.join('/')} content closes inline or on the previous line`,
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse, def)
        const passed = analysis.closingFormat !== 'other'
        return {
          passed,
          score: passed ? 1 : 0,
          message: analysis.closingFormat,
          snippet: analysis.primaryBody ?? undefined,
        }
      },
    },
    {
      id: 'no-prose',
      description: 'Check whether response contains only the tool call XML',
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse, def)
        return {
          passed: !analysis.hasProse,
          score: analysis.hasProse ? 0 : 1,
          message: analysis.hasProse ? 'has-prose' : 'no-prose',
          snippet: analysis.extraneousText || undefined,
        }
      },
    },
    {
      id: 'content-analysis',
      description: 'Extract raw body content and note trailing newline strategy',
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse, def)
        return {
          passed: analysis.primaryBody !== null,
          score: analysis.primaryBody !== null ? 1 : 0,
          message: [
            `body_format=${analysis.bodyFormat}`,
            `closing_format=${analysis.closingFormat}`,
            `trailing_newline_strategy=${analysis.trailingNewlineStrategy}`,
            `raw_body_length=${analysis.primaryBody?.length ?? 0}`,
          ].join('; '),
          snippet: analysis.primaryBody ?? undefined,
        }
      },
    },
  ]

  if (variant !== 'neutral') {
    checks.push({
      id: 'adherence',
      description: `Check whether the response follows the ${variant} formatting instruction`,
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse, def)
        const passed = variant === 'inline'
          ? analysis.bodyFormat === 'inline' && analysis.closingFormat === 'inline'
          : analysis.bodyFormat === 'next-line' && analysis.closingFormat === 'prev-line'
        return {
          passed,
          score: passed ? 1 : 0,
          message: passed ? `adheres-${variant}` : `violates-${variant}`,
          snippet: analysis.primaryBody ?? undefined,
        }
      },
    })
  }

  if (def.trailingNewlineFocus) {
    checks.push({
      id: 'newline-correctness',
      description: 'Check whether newline-sensitive content matches the requested trailing-newline behavior',
      evaluate(rawResponse: string): CheckResult {
        const analysis = analyzeResponse(rawResponse, def)
        const passed = analysis.correctness === 'correct'
        return {
          passed,
          score: passed ? 1 : 0,
          message: analysis.correctness,
          snippet: analysis.primaryBody ?? undefined,
        }
      },
    })
  }

  return checks
}

type BoundaryFormat = 'inline' | 'next-line' | 'prev-line' | 'other'
type TrailingStrategy = 'double-newline-before-close' | 'explicit-escape' | 'inline-with-newline' | 'no-attempt' | 'other'
type Correctness = 'correct' | 'incorrect' | 'missing-body'

interface TagCapture {
  tag: string
  inner: string
  fullMatch: string
  start: number
  end: number
}

interface ResponseAnalysis {
  bodyFormat: Exclude<BoundaryFormat, 'prev-line'>
  closingFormat: Exclude<BoundaryFormat, 'next-line'>
  hasProse: boolean
  extraneousText: string
  primaryBody: string | null
  trailingNewlineStrategy: TrailingStrategy
  correctness: Correctness
}

function analyzeResponse(rawResponse: string, def: XmlNewlineScenarioDef): ResponseAnalysis {
  const contentCaptures = def.group === 'write'
    ? extractTag(rawResponse, 'write')
    : extractTag(rawResponse, 'old').concat(extractTag(rawResponse, 'new'))
  const envelopeCaptures = extractTag(rawResponse, def.group === 'write' ? 'write' : 'edit')

  const primary = selectPrimaryCapture(contentCaptures, def)
  const matchedRanges = envelopeCaptures.map((capture) => [capture.start, capture.end] as const)
  const extraneousText = collectExtraneousText(rawResponse, matchedRanges)
  const primaryBody = primary?.inner ?? null

  return {
    bodyFormat: primary ? classifyBodyFormat(primary.inner) : 'other',
    closingFormat: primary ? classifyClosingFormat(primary.inner) : 'other',
    hasProse: extraneousText.trim().length > 0,
    extraneousText,
    primaryBody,
    trailingNewlineStrategy: classifyTrailingNewlineStrategy(primaryBody, def),
    correctness: classifyCorrectness(primaryBody, def),
  }
}

function selectPrimaryCapture(captures: TagCapture[], def: XmlNewlineScenarioDef): TagCapture | null {
  if (captures.length === 0) return null
  if (def.expectedPrimaryBody == null) return captures[0] ?? null
  return captures.find((capture) => normalizeForComparison(capture.inner) === def.expectedPrimaryBody) ?? captures[0] ?? null
}

function extractTag(raw: string, tag: string): TagCapture[] {
  const captures: TagCapture[] = []
  const regex = new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)</${tag}>`, 'gi')
  for (const match of raw.matchAll(regex)) {
    const fullMatch = match[0]
    const inner = match[1] ?? ''
    const start = match.index ?? 0
    captures.push({
      tag,
      inner,
      fullMatch,
      start,
      end: start + fullMatch.length,
    })
  }
  return captures
}

function collectExtraneousText(raw: string, ranges: ReadonlyArray<readonly [number, number]>): string {
  if (ranges.length === 0) return raw
  const sorted = [...ranges].sort((a, b) => a[0] - b[0])
  let cursor = 0
  let result = ''

  for (const [start, end] of sorted) {
    if (start > cursor) result += raw.slice(cursor, start)
    cursor = Math.max(cursor, end)
  }
  if (cursor < raw.length) result += raw.slice(cursor)
  return result
}

function classifyBodyFormat(inner: string): 'inline' | 'next-line' | 'other' {
  if (inner.startsWith('\n')) return 'next-line'
  if (inner.length > 0) return 'inline'
  return 'other'
}

function classifyClosingFormat(inner: string): 'inline' | 'prev-line' | 'other' {
  if (inner.endsWith('\n')) return 'prev-line'
  if (inner.length > 0) return 'inline'
  return 'other'
}

function classifyTrailingNewlineStrategy(body: string | null, def: XmlNewlineScenarioDef): TrailingStrategy {
  if (!def.trailingNewlineFocus || body == null) return 'no-attempt'
  if (body.includes('\\n')) return 'explicit-escape'
  if (body.endsWith('\n\n')) return 'double-newline-before-close'
  if (body.endsWith('\n')) return 'inline-with-newline'
  if (!body.includes('\n')) return 'no-attempt'
  return 'other'
}

function classifyCorrectness(body: string | null, def: XmlNewlineScenarioDef): Correctness {
  if (!def.trailingNewlineFocus) return body == null ? 'missing-body' : 'correct'
  if (body == null) return 'missing-body'

  const normalized = normalizeForComparison(body)
  const bodyMatches = def.expectedPrimaryBody == null || normalized === def.expectedPrimaryBody
  const trailingMatches = def.expectedTrailingNewline == null || normalized.endsWith('\n') === def.expectedTrailingNewline

  return bodyMatches && trailingMatches ? 'correct' : 'incorrect'
}

function normalizeForComparison(body: string): string {
  if (body.startsWith('\n') && body.endsWith('\n')) return body.slice(1, -1)
  return body
}

function parseScenarioId(id: string): { variant: VariantId; baseId: string } {
  const match = id.match(/^(neutral|inline|block)\/(.+)$/)
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

export const xmlNewlinesEval: RunnableEval = {
  id: 'xml-newlines',
  name: 'XML Newlines',
  description: `Measures XML tool-call newline behavior across ${variants.length} prompt variants and ${SCENARIO_DEFS.length} scenarios (${scenarios.length} total runs per model)`,
  scenarios,
  variants,
  defaultConcurrency: 4,
  defaultModels: DEFAULT_MODELS,

  async runScenario(scenario: Scenario, modelSpec: ModelSpec): Promise<ScenarioResult> {
    return executeScenario(scenario, modelSpec)
  },
}