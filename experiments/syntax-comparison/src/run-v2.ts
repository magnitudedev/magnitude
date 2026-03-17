import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { Effect, Layer, Stream } from 'effect'
import ansis from 'ansis'

import type { ChatMessage } from '@magnitudedev/llm-core'
import { ModelResolver, SimpleChat, createProviderClient, makeModelResolver, makeNoopTracer } from '@magnitudedev/providers'
import { scenarios } from './scenarios'
import type { ModelSpec, Scenario, Scores } from './types'

type Variation = 'bare' | 'wrapped' | 'typed' | 'xml-act'

interface VariationResult {
  model: string
  variation: Variation
  scenario: string
  raw_output: string
  scores: Scores
}

const VARIATIONS: Variation[] = ['bare', 'wrapped', 'typed', 'xml-act']

let cachedClientPromise: ReturnType<typeof createProviderClient> | null = null

async function getProviderClient() {
  if (!cachedClientPromise) cachedClientPromise = createProviderClient()
  return cachedClientPromise
}

function parseModelSpec(spec: string): ModelSpec {
  const colonIdx = spec.indexOf(':')
  if (colonIdx === -1) {
    throw new Error(`Invalid model spec "${spec}" -- expected "provider:model" format`)
  }
  const provider = spec.slice(0, colonIdx)
  const model = spec.slice(colonIdx + 1)
  return { provider, model, label: spec }
}

async function callModel(systemPrompt: string, messages: ChatMessage[], modelSpec: ModelSpec): Promise<string> {
  const client = await getProviderClient()
  const auth = await client.auth.getAuth(modelSpec.provider)
  await client.state.setSelection('primary', modelSpec.provider, modelSpec.model, auth ?? null, { persist: false })

  const chatStream = await Effect.runPromise(
    Effect.gen(function* () {
      const runtime = yield* ModelResolver
      const model = yield* runtime.resolve('primary')
      return yield* model.invoke(SimpleChat, { systemPrompt, messages })
    }).pipe(
      Effect.provide(
        Layer.merge(
          makeModelResolver().pipe(Layer.provide(client.layer), Layer.provide(makeNoopTracer())),
          makeNoopTracer(),
        ),
      ),
    ),
  )

  return Effect.runPromise(Stream.runFold(chatStream.stream, '', (acc, chunk) => acc + chunk))
}

function parseArgs(argv: string[]) {
  let model: string | null = null
  let scenarioId: string | null = null
  let concurrency = 4
  let ack = false
  let trials = 1

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--model') {
      model = argv[++i]
    } else if (arg === '--scenario') {
      scenarioId = argv[++i]
    } else if (arg === '--concurrency' || arg === '-n') {
      concurrency = parseInt(argv[++i], 10)
    } else if (arg === '--ack') {
      ack = true
    } else if (arg === '-t' || arg === '--trials') {
      trials = parseInt(argv[++i], 10)
    }
  }

  if (!model) throw new Error('Missing required --model provider:model')

  return { model, scenarioId, concurrency, ack, trials }
}

function loadPrompt(variation: Variation): string {
  const file =
    variation === 'bare'
      ? 'v2-bare.txt'
      : variation === 'wrapped'
        ? 'v2-wrapped.txt'
        : variation === 'typed'
          ? 'v2-typed.txt'
          : 'xml-act.txt'
  return readFileSync(resolve(import.meta.dir, 'prompts', file), 'utf8')
}

function selectScenarios(scenarioId: string | null): Scenario[] {
  if (!scenarioId) return scenarios
  return scenarios.filter(scenario => scenario.id === scenarioId)
}

function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, '')
}

function pad(s: string, len: number): string {
  const raw = stripAnsi(s)
  return s + ' '.repeat(Math.max(0, len - raw.length))
}

function scoreCell(v: boolean | null): string {
  if (v === null) return ansis.dim('·')
  return v ? ansis.green('✓') : ansis.red('✗')
}

function formatScoreBar(s: Scores): string {
  return [
    `syntax:${scoreCell(s.syntax_valid)}`,
    `1-turn:${scoreCell(s.single_turn)}`,
    `turn:${scoreCell(s.turn_control_correct)}`,
    `pd:${scoreCell(s.pd_used)}`,
    `halluc:${scoreCell(s.no_hallucinated_results)}`,
    `escape:${scoreCell(s.escaping_correct)}`,
  ].join('  ')
}

function printRunningStats(results: VariationResult[]) {
  const VARIATIONS: Variation[] = ['bare', 'wrapped', 'typed', 'xml-act']
  const criteriaKeys: { key: keyof Scores; label: string }[] = [
    { key: 'syntax_valid', label: 'syntax' },
    { key: 'single_turn', label: '1-turn' },
    { key: 'turn_control_correct', label: 'turn' },
    { key: 'pd_used', label: 'pd' },
    { key: 'no_hallucinated_results', label: 'halluc' },
    { key: 'escaping_correct', label: 'escape' },
  ]

  const totalWidth = process.stdout.columns || 120
  const colWidth = Math.floor((totalWidth - 6) / 3)

  let header = ''
  for (const v of VARIATIONS) {
    header += pad(ansis.bold(v.toUpperCase()), colWidth + 2)
  }
  console.log(ansis.dim('  ' + header))

  let statsLine = '  '
  for (const v of VARIATIONS) {
    const vResults = results.filter(r => r.variation === v)
    if (vResults.length === 0) {
      statsLine += pad(ansis.dim('—'), colWidth + 2)
      continue
    }
    const parts: string[] = []
    for (const { key, label } of criteriaKeys) {
      const applicable = vResults.filter(r => r.scores[key] !== null)
      if (applicable.length === 0) continue
      const passed = applicable.filter(r => r.scores[key] === true).length
      const pct = Math.round((passed / applicable.length) * 100)
      const color = pct === 100 ? ansis.green : pct >= 75 ? ansis.yellow : ansis.red
      parts.push(`${label}:${color(pct + '%')}`)
    }
    statsLine += pad(parts.join(' '), colWidth + 2)
  }
  console.log(statsLine)
  console.log('')
}

function wrapText(text: string, width: number): string[] {
  const lines: string[] = []
  for (const raw of text.split('\n')) {
    if (stripAnsi(raw).length <= width) {
      lines.push(raw)
    } else {
      let remaining = raw
      while (stripAnsi(remaining).length > width) {
        lines.push(remaining.slice(0, width))
        remaining = remaining.slice(width)
      }
      if (remaining) lines.push(remaining)
    }
  }
  return lines
}

const COLUMN_COLORS = [ansis.yellow, ansis.magenta, ansis.cyan, ansis.green]

function printColumns(columns: { title: string; body: string }[]) {
  const totalWidth = process.stdout.columns || 180
  const numCols = columns.length
  const colWidth = Math.floor((totalWidth - (numCols - 1) * 3) / numCols)
  const sep = ansis.dim(' │ ')

  const allLines = columns.map(c => wrapText(c.body, colWidth))
  const maxLines = Math.max(...allLines.map(l => l.length))

  // Headers
  console.log(columns.map((c, i) => {
    const color = COLUMN_COLORS[i % COLUMN_COLORS.length]
    return i < numCols - 1 ? color.bold(pad(c.title, colWidth)) : color.bold(c.title)
  }).join(sep))

  console.log(columns.map((_, i) =>
    i < numCols - 1 ? ansis.dim('─'.repeat(colWidth)) : ansis.dim('─'.repeat(colWidth))
  ).join(sep))

  for (let i = 0; i < maxLines; i++) {
    console.log(columns.map((_, ci) => {
      const line = allLines[ci][i] ?? ''
      return ci < numCols - 1 ? pad(line, colWidth) : line
    }).join(sep))
  }
}

function hasSingleLensesBlock(output: string): boolean {
  return (output.match(/<lenses>/g) ?? []).length === 1
}

function countObserveAttrs(output: string): number {
  return (output.match(/\bobserve="[^"]*"/g) ?? []).length
}

function usesPd(variation: Variation, output: string): boolean {
  if (variation === 'xml-act') {
    return /<inspect>[\s\S]*?<ref\b/.test(output)
  }
  return countObserveAttrs(output) > 0
}

function xmlActSyntax(output: string): boolean {
  if (!/^\s*<reasoning>/.test(output)) return false
  if (!/<(next|yield)\s*\/>\s*$/.test(output)) return false
  const hasReasoning = /<reasoning>[\s\S]*?<\/reasoning>/.test(output)
  const hasActions = /<actions[\s\S]*?<\/actions>/.test(output) || /<actions\s*\/>/.test(output)
  return hasReasoning && hasActions
}

function bareSyntax(output: string): boolean {
  if (!hasSingleLensesBlock(output)) return false
  if (/<turn(?::\w+)?>/.test(output)) return false
  // Must start with <lenses> (after optional whitespace)
  if (!/^\s*<lenses>/.test(output)) return false

  const hasDeclare = /<declare>[\s\S]*?<\/declare>/.test(output)
  const hasObserve = /<observe\s*\/>/.test(output)
  const hasConclude = /<conclude>[\s\S]*?<\/conclude>/.test(output)

  if (hasDeclare) return hasObserve || hasConclude
  return hasConclude
}

function wrappedSyntax(output: string): boolean {
  const turns = output.match(/<turn>/g) ?? []
  const closes = output.match(/<\/turn>/g) ?? []
  if (turns.length !== 1 || closes.length !== 1) return false
  if (!hasSingleLensesBlock(output)) return false

  const match = output.match(/^(\s*)<turn>[\s\S]*<\/turn>\s*$/)
  if (!match) return false

  const hasDeclare = /<declare>[\s\S]*?<\/declare>/.test(output)
  const hasObserve = /<observe\s*\/>/.test(output)
  const hasConclude = /<conclude>[\s\S]*?<\/conclude>/.test(output)

  if (hasDeclare) return hasObserve || hasConclude
  return hasConclude
}

function typedSyntax(output: string): boolean {
  if (!hasSingleLensesBlock(output)) return false
  if (/<declare>|<observe\s*\/>|<conclude>/.test(output)) return false

  const workCount = (output.match(/<turn:work>/g) ?? []).length
  const askCount = (output.match(/<turn:ask>/g) ?? []).length
  const answerCount = (output.match(/<turn:answer>/g) ?? []).length
  const total = workCount + askCount + answerCount
  if (total !== 1) return false

  return /^(\s*)<(turn:work|turn:ask|turn:answer)>[\s\S]*<\/(turn:work|turn:ask|turn:answer)>\s*$/.test(output)
}

function isSingleTurn(variation: Variation, output: string): boolean {
  if (variation === 'xml-act') {
    return (output.match(/<reasoning>/g) ?? []).length <= 1
  }
  return hasSingleLensesBlock(output)
}

function continues(variation: Variation, output: string): boolean {
  if (variation === 'xml-act') return /<next\s*\/>/.test(output)
  if (variation === 'typed') return /<turn:work>/.test(output)
  return /<observe\s*\/>/.test(output)
}

function extractUserFacingText(variation: Variation, output: string): string {
  if (variation === 'xml-act') {
    return output.match(/<message>([\s\S]*?)<\/message>/)?.[1]?.trim() ?? ''
  }
  if (variation === 'typed') {
    const askText = output.match(/<turn:ask>[\s\S]*?<\/lenses>([\s\S]*?)<\/turn:ask>/)?.[1]?.trim() ?? ''
    const answerText = output.match(/<turn:answer>[\s\S]*?<\/lenses>([\s\S]*?)<\/turn:answer>/)?.[1]?.trim() ?? ''
    return `${askText} ${answerText}`.trim()
  }

  const concludeText = output.match(/<conclude>([\s\S]*?)<\/conclude>/)?.[1] ?? ''
  return concludeText.trim()
}

function hallucinatesResults(variation: Variation, output: string): boolean {
  const userText = extractUserFacingText(variation, output)
  const claimsResults = /\b(found|contains|shows|output|passed|failed|exit code \d|error|TODO|host|localhost|the file|the config|results?:|here'?s what)/i.test(userText)
  const hasReadOrSearchOrRun = /\b(read|search|run)\b/.test(output)
  return claimsResults && hasReadOrSearchOrRun && !continues(variation, output)
}

function stripCodeFences(output: string): string {
  let s = output.trim()
  if (s.startsWith('```')) {
    s = s.replace(/^```[^\n]*\n?/, '').replace(/\n?```\s*$/, '')
  }
  return s.trim()
}

function hasObserveAndConclude(output: string): boolean {
  return /<observe\s*\/>/.test(output) && /<conclude>[\s\S]*?<\/conclude>/.test(output)
}

function evaluateVariation(variation: Variation, scenario: Scenario, rawOutput: string): Scores {
  rawOutput = stripCodeFences(rawOutput)

  const syntax_valid =
    variation === 'xml-act'
      ? xmlActSyntax(rawOutput)
      : variation === 'bare'
        ? bareSyntax(rawOutput)
        : variation === 'wrapped'
          ? wrappedSyntax(rawOutput)
          : typedSyntax(rawOutput)

  // Turn control checks
  let turn_control_correct = true

  // If observe + conclude coexist in same response, that's wrong
  if (hasObserveAndConclude(rawOutput)) {
    turn_control_correct = false
  }
  // If should continue but didn't, wrong if hallucinated
  else if (scenario.shouldContinue && !continues(variation, rawOutput)) {
    turn_control_correct = !hallucinatesResults(variation, rawOutput)
  }

  // no-tools scenario: using tools is wrong turn control
  else if (scenario.id === 'no-tools' && /<(read|write|edit|search|run|tree)\b/.test(rawOutput)) {
    turn_control_correct = false
  }

  // Hallucination: also catch observe+conclude pattern
  const hallucinated = hallucinatesResults(variation, rawOutput) || hasObserveAndConclude(rawOutput)

  return {
    syntax_valid,
    single_turn: isSingleTurn(variation, rawOutput),
    turn_control_correct,
    pd_used: scenario.shouldUsePd !== null ? usesPd(variation, rawOutput) : null,
    no_hallucinated_results: scenario.checkNoHallucination !== null ? !hallucinated : null,
    escaping_correct: scenario.checkEscaping != null ? rawOutput.includes(scenario.checkEscaping) : null,
  }
}

function buildAckTurn(variation: Variation): ChatMessage[] {
  if (variation === 'bare') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['<lenses>\n<lens name="intent">Acknowledge format.</lens>\n<lens name="state">Need only a syntax demo.</lens>\n<lens name="strategy">Show one work turn.</lens>\n</lenses>\n<declare>Demonstrating the format.</declare>\n<read path="example.txt" observe=".content"/>\n<observe/>'],
      },
    ]
  }

  if (variation === 'wrapped') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['<turn>\n<lenses>\n<lens name="intent">Acknowledge format.</lens>\n<lens name="state">Need only a syntax demo.</lens>\n<lens name="strategy">Show one wrapped turn.</lens>\n</lenses>\n<declare>Demonstrating the format.</declare>\n<read path="example.txt" observe=".content"/>\n<observe/>\n</turn>'],
      },
    ]
  }

  if (variation === 'xml-act') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['<reasoning>Understood.</reasoning>\n<message>Ready.</message>\n<actions/>\n<yield/>'],
      },
    ]
  }

  return [
    { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
    {
      role: 'assistant',
      content: ['<turn:answer>\n<lenses>\n<lens name="intent">Acknowledge format.</lens>\n<lens name="state">Need only a syntax demo.</lens>\n<lens name="strategy">Reply directly.</lens>\n</lenses>\nReady.\n</turn:answer>'],
    },
  ]
}

async function runScenarioTriple(modelSpec: ModelSpec, scenario: Scenario, ack: boolean): Promise<VariationResult[]> {
  const results = await Promise.all(
    VARIATIONS.map(async (variation) => {
      const systemPrompt = loadPrompt(variation)
      const messages: ChatMessage[] = [
        ...(ack ? buildAckTurn(variation) : []),
        { role: 'user', content: [scenario.userMessage] },
      ]

      try {
        const raw_output = await callModel(systemPrompt, messages, modelSpec)
        return {
          model: modelSpec.label,
          variation,
          scenario: scenario.id,
          raw_output,
          scores: evaluateVariation(variation, scenario, raw_output),
        } satisfies VariationResult
      } catch (err) {
        console.error(ansis.red(`  ✗ ${variation}: ${err}`))
        return null
      }
    }),
  )

  return results.filter((r): r is VariationResult => r !== null)
}

function aggregateScores(results: VariationResult[]): { passed: number; total: number } {
  let passed = 0
  let total = 0
  for (const r of results) {
    for (const v of Object.values(r.scores)) {
      if (v !== null) {
        total++
        if (v) passed++
      }
    }
  }
  return { passed, total }
}

function printSummary(results: VariationResult[]) {
  const scenarioIds = [...new Set(results.map(r => r.scenario))]
  const criteria = ['syntax', '1-turn', 'turn', 'pd', 'halluc', 'escape'] as const
  const labels = { syntax: 'Syntax', '1-turn': '1 Turn', turn: 'Turn Ctl', pd: 'Prog Disc', halluc: 'No Halluc', escape: 'Escaping' }
  const scoreKeys: Record<typeof criteria[number], keyof Scores> = {
    syntax: 'syntax_valid',
    '1-turn': 'single_turn',
    turn: 'turn_control_correct',
    pd: 'pd_used',
    halluc: 'no_hallucinated_results',
    escape: 'escaping_correct',
  }

  console.log('')
  console.log(ansis.bold.underline('Summary'))
  console.log('')

  const scenarioCol = 18
  const cellWidth = 10

  let header = pad('', scenarioCol)
  for (const variation of VARIATIONS) {
    header += ansis.bold(pad(variation, criteria.length * cellWidth))
  }
  console.log(header)

  let subheader = pad('Scenario', scenarioCol)
  for (const _variation of VARIATIONS) {
    for (const c of criteria) {
      subheader += ansis.dim(pad(labels[c], cellWidth))
    }
  }
  console.log(subheader)
  console.log(ansis.dim('─'.repeat(scenarioCol + VARIATIONS.length * criteria.length * cellWidth)))

  for (const scenarioId of scenarioIds) {
    let row = pad(scenarioId, scenarioCol)
    for (const variation of VARIATIONS) {
      const result = results.find(r => r.scenario === scenarioId && r.variation === variation)
      if (result) {
        for (const c of criteria) {
          row += pad(scoreCell(result.scores[scoreKeys[c]]), cellWidth)
        }
      } else {
        for (const _c of criteria) {
          row += pad(ansis.dim('?'), cellWidth)
        }
      }
    }
    console.log(row)
  }

  console.log(ansis.dim('─'.repeat(scenarioCol + VARIATIONS.length * criteria.length * cellWidth)))
  let aggRow = pad(ansis.bold('Score'), scenarioCol)
  for (const variation of VARIATIONS) {
    const variationResults = results.filter(r => r.variation === variation)
    const { passed, total } = aggregateScores(variationResults)
    const pct = total > 0 ? Math.round((passed / total) * 100) : 0
    const color = pct === 100 ? ansis.green : pct >= 75 ? ansis.yellow : ansis.red
    aggRow += pad(color(`${passed}/${total} (${pct}%)`), criteria.length * cellWidth)
  }
  console.log(aggRow)
  console.log('')
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const modelSpec = parseModelSpec(args.model)
  const chosenScenarios = selectScenarios(args.scenarioId)

  if (chosenScenarios.length === 0) {
    throw new Error(args.scenarioId ? `Unknown scenario: ${args.scenarioId}` : 'No scenarios selected')
  }

  const trials = args.trials
  console.log(ansis.bold(`Running ${chosenScenarios.length} scenarios × 3 variations × ${trials} trial(s) for ${modelSpec.label} with concurrency ${args.concurrency}\n`))

  const allResults: VariationResult[] = []

  interface TrialTask {
    scenario: Scenario
    trial: number
  }

  const tasks: TrialTask[] = []
  for (const scenario of chosenScenarios) {
    for (let t = 1; t <= trials; t++) {
      tasks.push({ scenario, trial: t })
    }
  }

  let idx = 0

  async function worker() {
    while (idx < tasks.length) {
      const i = idx++
      const { scenario, trial } = tasks[i]

      const trialLabel = trials > 1 ? ` (trial ${trial}/${trials})` : ''
      console.log(ansis.bold.cyan(`\n── ${scenario.id}${trialLabel} ──`))
      console.log(ansis.dim(`   ${scenario.description}`))
      console.log(ansis.dim(`   User: "${scenario.userMessage}"`))
      console.log(ansis.dim(`   Expected: continue=${scenario.shouldContinue ? 'yes' : 'no'}, pd=${scenario.shouldUsePd === null ? 'n/a' : scenario.shouldUsePd ? 'yes' : 'no'}`))

      const triple = await runScenarioTriple(modelSpec, scenario, args.ack)
      allResults.push(...triple)

      console.log('')
      printColumns(
        VARIATIONS.map(v => {
          const r = triple.find(r => r.variation === v)
          return {
            title: v.toUpperCase(),
            body: r ? `${r.raw_output}\n\n${formatScoreBar(r.scores)}` : ansis.dim('(not run)'),
          }
        }),
      )

      // Print running percentages
      printRunningStats(allResults)
    }
  }

  const workers = Array.from({ length: Math.min(args.concurrency, tasks.length) }, () => worker())
  await Promise.all(workers)

  printSummary(allResults)

  // Auto-save results
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const outDir = resolve(import.meta.dir, '..', 'results')
  const { mkdirSync, writeFileSync } = await import('node:fs')
  mkdirSync(outDir, { recursive: true })
  const outPath = resolve(outDir, `v2-${timestamp}.json`)
  writeFileSync(outPath, JSON.stringify({ timestamp: new Date().toISOString(), model: modelSpec.label, trials: args.trials, results: allResults }, null, 2))
  console.log(ansis.dim(`\nResults saved to ${outPath}`))
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})