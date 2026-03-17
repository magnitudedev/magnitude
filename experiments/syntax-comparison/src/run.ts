import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { Effect, Layer, Stream } from 'effect'
import ansis from 'ansis'

import type { ChatMessage } from '@magnitudedev/llm-core'
import { ModelResolver, SimpleChat, createProviderClient, makeModelResolver, makeNoopTracer } from '@magnitudedev/providers'
import { evaluateOutput } from './evaluate'
import { scenarios } from './scenarios'
import type { Format, ModelSpec, Result, RunOutput, Scenario, Scores } from './types'

const DEFAULT_MODELS: ModelSpec[] = [
  { provider: 'anthropic', model: 'claude-sonnet-4-6', label: 'anthropic:claude-sonnet-4-6' },
  { provider: 'anthropic', model: 'claude-haiku-3-5', label: 'anthropic:claude-haiku-3-5' },
  { provider: 'openai', model: 'gpt-4o-mini', label: 'openai:gpt-4o-mini' },
  { provider: 'openai', model: 'gpt-4o', label: 'openai:gpt-4o' },
]

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
  const models: string[] = []
  let format: Format | null = null
  let scenarioId: string | null = null
  let output: string | null = null

  let concurrency = 4
  let ack = false
  let xmlV2 = false
  let strOpen = '#['
  let strClose = ']#'

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--model') {
      models.push(argv[++i])
    } else if (arg === '--format') {
      format = argv[++i] as Format
    } else if (arg === '--scenario') {
      scenarioId = argv[++i]
    } else if (arg === '--output') {
      output = argv[++i]
    } else if (arg === '--concurrency' || arg === '-n') {
      concurrency = parseInt(argv[++i], 10)
    } else if (arg === '--ack') {
      ack = true
    } else if (arg === '--xml-v2') {
      xmlV2 = true
    } else if (arg === '--open') {
      strOpen = argv[++i]
    } else if (arg === '--close') {
      strClose = argv[++i]
    }
  }

  return { models, format, scenarioId, output, concurrency, ack, strOpen, strClose, xmlV2 }
}

function loadPrompt(format: Format, strOpen: string, strClose: string): string {
  const file = format === 'xml-act' ? 'xml-act.txt' : format === 'xml-v2' ? 'xml-v2.txt' : 'declare.txt'
  let prompt = readFileSync(resolve(import.meta.dir, 'prompts', file), 'utf8')
  if (format === 'declare') {
    prompt = prompt.replaceAll('#[', strOpen).replaceAll(']#', strClose)
  }
  return prompt
}

function selectScenarios(scenarioId: string | null): Scenario[] {
  if (!scenarioId) return scenarios
  return scenarios.filter(scenario => scenario.id === scenarioId)
}

function scoreCell(v: boolean | null): string {
  if (v === null) return ansis.dim('·')
  return v ? ansis.green('✓') : ansis.red('✗')
}

function pad(s: string, len: number): string {
  // Strip ANSI for length calculation
  const raw = s.replace(/\x1b\[[0-9;]*m/g, '')
  return s + ' '.repeat(Math.max(0, len - raw.length))
}

function aggregateScores(results: Result[]): { passed: number; total: number } {
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

function printSummary(results: Result[]) {
  // Group by model
  const models = [...new Set(results.map(r => r.model))]
  const formats = [...new Set(results.map(r => r.format))] as Format[]
  const scenarioIds = [...new Set(results.map(r => r.scenario))]

  const criteria = ['syntax', '1-turn', 'turn', 'pd', 'halluc', 'escape'] as const
  const criteriaLabels = { syntax: 'Syntax', '1-turn': '1 Turn', turn: 'Turn Ctl', pd: 'Prog Disc', halluc: 'No Halluc', escape: 'Escaping' }
  const scoreKeys: Record<typeof criteria[number], keyof Scores> = {
    syntax: 'syntax_valid',
    '1-turn': 'single_turn',
    turn: 'turn_control_correct',
    pd: 'pd_used',
    halluc: 'no_hallucinated_results',
    escape: 'escaping_correct',
  }

  for (const model of models) {
    console.log('')
    console.log(ansis.bold.underline(model))
    console.log('')

    // Header
    const scenarioCol = 18
    const cellWidth = 10
    let header = pad('', scenarioCol)
    for (const format of formats) {
      header += ansis.bold(pad(format, criteria.length * cellWidth))
    }
    console.log(header)

    let subheader = pad('Scenario', scenarioCol)
    for (const _format of formats) {
      for (const c of criteria) {
        subheader += ansis.dim(pad(criteriaLabels[c], cellWidth))
      }
    }
    console.log(subheader)
    console.log(ansis.dim('─'.repeat(scenarioCol + formats.length * criteria.length * cellWidth)))

    for (const scenarioId of scenarioIds) {
      let row = pad(scenarioId, scenarioCol)
      for (const format of formats) {
        const result = results.find(r => r.model === model && r.format === format && r.scenario === scenarioId)
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

    // Per-format aggregate
    console.log(ansis.dim('─'.repeat(scenarioCol + formats.length * criteria.length * cellWidth)))
    let aggRow = pad(ansis.bold('Score'), scenarioCol)
    for (const format of formats) {
      const formatResults = results.filter(r => r.model === model && r.format === format)
      const { passed, total } = aggregateScores(formatResults)
      const pct = total > 0 ? Math.round((passed / total) * 100) : 0
      const color = pct === 100 ? ansis.green : pct >= 75 ? ansis.yellow : ansis.red
      const scoreStr = color(`${passed}/${total} (${pct}%)`)
      aggRow += pad(scoreStr, criteria.length * cellWidth)
    }
    console.log(aggRow)
  }

  // Overall comparison if multiple models
  if (models.length > 1) {
    console.log('')
    console.log(ansis.bold.underline('Overall by Format'))
    console.log('')
    for (const format of formats) {
      const formatResults = results.filter(r => r.format === format)
      const { passed, total } = aggregateScores(formatResults)
      const pct = total > 0 ? Math.round((passed / total) * 100) : 0
      const color = pct === 100 ? ansis.green : pct >= 75 ? ansis.yellow : ansis.red
      console.log(`  ${ansis.bold(pad(format, 12))} ${color(`${passed}/${total} (${pct}%)`)}`)
    }
  }

  console.log('')
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

function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, '')
}

function wrapText(text: string, width: number): string[] {
  const lines: string[] = []
  for (const raw of text.split('\n')) {
    if (stripAnsi(raw).length <= width) {
      lines.push(raw)
    } else {
      // Hard wrap long lines
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

function printSideBySide(leftTitle: string, leftBody: string, rightTitle: string, rightBody: string) {
  const totalWidth = process.stdout.columns || 120
  const colWidth = Math.floor((totalWidth - 3) / 2) // 3 for separator " │ "
  const sep = ansis.dim(' │ ')

  const leftLines = wrapText(leftBody, colWidth)
  const rightLines = wrapText(rightBody, colWidth)
  const maxLines = Math.max(leftLines.length, rightLines.length)

  // Headers
  console.log(
    ansis.yellow.bold(pad(leftTitle, colWidth)) + sep + ansis.magenta.bold(rightTitle)
  )
  console.log(ansis.dim('─'.repeat(colWidth)) + sep + ansis.dim('─'.repeat(colWidth)))

  for (let i = 0; i < maxLines; i++) {
    const left = leftLines[i] ?? ''
    const right = rightLines[i] ?? ''
    console.log(pad(left, colWidth) + sep + right)
  }
}

function printComparisonByScenario(results: Result[]) {
  const models = [...new Set(results.map(r => r.model))]
  const scenarioIds = [...new Set(results.map(r => r.scenario))]
  const allScenarios = scenarios

  for (const model of models) {
    console.log('')
    console.log(ansis.bold.underline(`Model: ${model}`))

    for (const scenarioId of scenarioIds) {
      const scenario = allScenarios.find(s => s.id === scenarioId)
      const xmlResult = results.find(r => r.model === model && r.format === 'xml-act' && r.scenario === scenarioId)
      const dslResult = results.find(r => r.model === model && r.format === 'declare' && r.scenario === scenarioId)

      console.log('')
      console.log(ansis.bold.cyan(`── ${scenarioId} ──`))
      console.log(ansis.dim(`   ${scenario?.description ?? ''}`))
      console.log(ansis.dim(`   User: "${scenario?.userMessage ?? ''}"`))
      console.log(ansis.dim(`   Expected: continue=${scenario?.shouldContinue ? 'yes' : 'no'}, pd=${scenario?.shouldUsePd === null ? 'n/a' : scenario?.shouldUsePd ? 'yes' : 'no'}`))
      console.log('')

      const xmlBody = xmlResult
        ? xmlResult.raw_output + '\n\n' + formatScoreBar(xmlResult.scores)
        : ansis.dim('(not run)')
      const dslBody = dslResult
        ? dslResult.raw_output + '\n\n' + formatScoreBar(dslResult.scores)
        : ansis.dim('(not run)')

      printSideBySide('XML-ACT', xmlBody, 'DECLARE-OBSERVE', dslBody)
    }
  }
}

function buildAckTurn(format: Format, strOpen: string, strClose: string): ChatMessage[] {
  if (format === 'xml-act') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      { role: 'assistant', content: ['<reasoning>Understood.</reasoning>\n<message>Ready.</message>\n<actions/>\n<yield/>'] },
    ]
  }
  if (format === 'xml-v2') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax. Use a declare/observe block with a mock read action.'] },
      { role: 'assistant', content: ['<turn:work>\n<lenses>\n<lens name="intent">Demonstrating format.</lens>\n</lenses>\n<declare>Checking syntax.</declare>\n<read path="example.txt" observe=".content"/>\n<observe/>\n</turn:work>'] },
    ]
  }
  return [
    { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax. Use a declare/do/observe block with a mock read action.'] },
    { role: 'assistant', content: [`think ${strOpen}\nDemonstrating format understanding.\n${strClose}\n\ndeclare ${strOpen}Checking syntax.${strClose} do {\n  (read file=${strOpen}example.txt${strClose})[.content] -> @\n} observe @`] },
  ]
}

async function runScenarioPair(
  modelSpec: ModelSpec,
  scenario: Scenario,
  formats: Format[],
  concurrency: number,
  ack: boolean,
  strOpen: string,
  strClose: string,
): Promise<Result[]> {
  const results = await Promise.all(
    formats.map(async (format) => {
      const systemPrompt = loadPrompt(format, strOpen, strClose)
      console.log(ansis.dim(`  ▸ ${format} ...`))
      const messages: ChatMessage[] = [
        ...(ack ? buildAckTurn(format, strOpen, strClose) : []),
        { role: 'user', content: [scenario.userMessage] },
      ]
      try {
        const raw_output = await callModel(
          systemPrompt,
          messages,
          modelSpec,
        )
        return {
          model: modelSpec.label,
          format,
          scenario: scenario.id,
          raw_output,
          scores: evaluateOutput(format, scenario, raw_output, strOpen, strClose),
        } as Result
      } catch (err) {
        console.error(ansis.red(`  ✗ ${format}: ${err}`))
        return null
      }
    }),
  )
  return results.filter((r): r is Result => r !== null)
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const modelSpecs = args.models.length > 0 ? args.models.map(parseModelSpec) : DEFAULT_MODELS
  const xmlFormat: Format = args.xmlV2 ? 'xml-v2' : 'xml-act'
  const formats: Format[] = args.format ? [args.format] : [xmlFormat, 'declare']
  const chosenScenarios = selectScenarios(args.scenarioId)

  if (args.scenarioId && chosenScenarios.length === 0) {
    throw new Error(`Unknown scenario: ${args.scenarioId}`)
  }

  const totalPairs = modelSpecs.length * chosenScenarios.length
  console.log(ansis.bold(`Running ${totalPairs} scenario pairs (${formats.join(' vs ')}) with concurrency ${args.concurrency}\n`))

  const allResults: Result[] = []

  for (const modelSpec of modelSpecs) {
    console.log(ansis.bold.underline(`\nModel: ${modelSpec.label}`))

    // Run scenarios with concurrency
    let idx = 0
    const scenarioQueue = [...chosenScenarios]

    async function worker() {
      while (idx < scenarioQueue.length) {
        const i = idx++
        const scenario = scenarioQueue[i]

        console.log(ansis.bold.cyan(`\n── ${scenario.id} ──`))
        console.log(ansis.dim(`   ${scenario.description}`))
        console.log(ansis.dim(`   User: "${scenario.userMessage}"`))
        console.log(ansis.dim(`   Expected: continue=${scenario.shouldContinue ? 'yes' : 'no'}, pd=${scenario.shouldUsePd === null ? 'n/a' : scenario.shouldUsePd ? 'yes' : 'no'}`))

        const pairResults = await runScenarioPair(modelSpec, scenario, formats, args.concurrency, args.ack, args.strOpen, args.strClose)
        allResults.push(...pairResults)

        // Print side-by-side immediately
        const xmlResult = pairResults.find(r => r.format === 'xml-act' || r.format === 'xml-v2')
        const dslResult = pairResults.find(r => r.format === 'declare')

        if (xmlResult && dslResult) {
          const xmlLabel = xmlResult.format === 'xml-v2' ? 'XML-V2' : 'XML-ACT'
          console.log('')
          printSideBySide(
            xmlLabel,
            xmlResult.raw_output + '\n\n' + formatScoreBar(xmlResult.scores),
            'DECLARE-OBSERVE',
            dslResult.raw_output + '\n\n' + formatScoreBar(dslResult.scores),
          )
        } else {
          // Single format mode
          for (const r of pairResults) {
            console.log('')
            console.log(ansis.bold(`  ${r.format}`))
            console.log(r.raw_output)
            console.log(formatScoreBar(r.scores))
          }
        }
      }
    }

    const workers = Array.from(
      { length: Math.min(args.concurrency, scenarioQueue.length) },
      () => worker(),
    )
    await Promise.all(workers)
  }

  printSummary(allResults)

  if (args.output) {
    const payload: RunOutput = {
      timestamp: new Date().toISOString(),
      results,
    }
    writeFileSync(args.output, JSON.stringify(payload, null, 2))
    console.log(`\nSaved results to ${args.output}`)
  }
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})