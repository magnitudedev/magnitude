import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { Effect, Layer, Stream } from 'effect'
import ansis from 'ansis'

import type { ChatMessage } from '@magnitudedev/llm-core'
import { ModelResolver, SimpleChat, createProviderClient, makeModelResolver, makeNoopTracer } from '@magnitudedev/providers'
import { scenarios } from './scenarios'
import type { ModelSpec, Scenario, Scores } from './types'

type Variation = 'xml-act' | 'flat-v4'

interface VariationResult {
  model: string
  variation: Variation
  scenario: string
  raw_output: string
  scores: Scores
}

const VARIATIONS: Variation[] = ['xml-act', 'flat-v4']

let cachedClientPromise: ReturnType<typeof createProviderClient> | null = null

async function getProviderClient() {
  if (!cachedClientPromise) cachedClientPromise = createProviderClient({ slots: ['primary'] as const })
  return cachedClientPromise
}

function parseModelSpec(spec: string): ModelSpec {
  const colonIdx = spec.indexOf(':')
  if (colonIdx === -1) throw new Error(`Invalid model spec "${spec}" -- expected "provider:model" format`)
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
    if (arg === '--model') model = argv[++i]
    else if (arg === '--scenario') scenarioId = argv[++i]
    else if (arg === '--concurrency' || arg === '-n') concurrency = parseInt(argv[++i], 10)
    else if (arg === '--ack') ack = true
    else if (arg === '-t' || arg === '--trials') trials = parseInt(argv[++i], 10)
  }

  if (!model) throw new Error('Missing required --model provider:model')
  return { model, scenarioId, concurrency, ack, trials }
}

function loadPrompt(variation: Variation): string {
  const file = variation === 'xml-act' ? 'xml-act.txt' : 'flat-v4.txt'
  return readFileSync(resolve(import.meta.dir, 'prompts', file), 'utf8')
}

function selectScenarios(scenarioId: string | null): Scenario[] {
  if (!scenarioId) return scenarios
  return scenarios.filter(s => s.id === scenarioId)
}

function stripAnsi(s: string): string {
  return s.replace(/\x1b\[[0-9;]*m/g, '')
}

function pad(s: string, len: number): string {
  return s + ' '.repeat(Math.max(0, len - stripAnsi(s).length))
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

function wrapText(text: string, width: number): string[] {
  const lines: string[] = []
  for (const raw of text.split('\n')) {
    if (stripAnsi(raw).length <= width) lines.push(raw)
    else {
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

function printColumns(columns: { title: string; body: string }[]) {
  const totalWidth = process.stdout.columns || 180
  const numCols = columns.length
  const colWidth = Math.floor((totalWidth - (numCols - 1) * 3) / numCols)
  const sep = ansis.dim(' │ ')
  const allLines = columns.map(c => wrapText(c.body, colWidth))
  const maxLines = Math.max(...allLines.map(l => l.length))
  const colors = [ansis.yellow, ansis.magenta]

  console.log(columns.map((c, i) => colors[i % colors.length].bold(i < numCols - 1 ? pad(c.title, colWidth) : c.title)).join(sep))
  console.log(columns.map(() => ansis.dim('─'.repeat(colWidth))).join(sep))

  for (let i = 0; i < maxLines; i++) {
    console.log(columns.map((_, ci) => {
      const line = allLines[ci][i] ?? ''
      return ci < numCols - 1 ? pad(line, colWidth) : line
    }).join(sep))
  }
}

function hasObserveAttr(content: string): boolean {
  return /\bobserve="[^"]*"/.test(content)
}

function hasAnyTool(content: string): boolean {
  return /<(read|write|edit|grep|run|tree|read|write|fs-edit|grep|shell|tree)\b/.test(content)
}

function stripCodeFences(output: string): string {
  let s = output.trim()
  if (s.startsWith('```')) s = s.replace(/^```[^\n]*\n?/, '').replace(/\n?```\s*$/, '')
  return s.trim()
}

type XmlActBlock =
  | { type: 'think'; content: string }
  | { type: 'comms'; content: string }
  | { type: 'actions'; content: string }
  | { type: 'next' }
  | { type: 'yield' }
  | { type: 'unknown'; content: string }

function extractXmlActBlocks(output: string): XmlActBlock[] {
  const blocks: XmlActBlock[] = []
  let pos = 0
  while (pos < output.length) {
    while (pos < output.length && /\s/.test(output[pos])) pos++
    if (pos >= output.length) break

    if (output.startsWith('<lenses>', pos)) {
      const close = output.indexOf('</lenses>', pos)
      if (close === -1) return [...blocks, { type: 'unknown', content: output.slice(pos) }]
      const inner = output.slice(pos + '<lenses>'.length, close)
      const lenses = [...inner.matchAll(/<lens\s+name="[^"]+">([\s\S]*?)<\/lens>/g)]
      if (lenses.length === 0) blocks.push({ type: 'unknown', content: inner })
      else for (const l of lenses) blocks.push({ type: 'think', content: l[1] ?? '' })
      pos = close + '</lenses>'.length
      continue
    }

    if (output.startsWith('<comms>', pos)) {
      const close = output.indexOf('</comms>', pos)
      if (close === -1) return [...blocks, { type: 'unknown', content: output.slice(pos) }]
      blocks.push({ type: 'comms', content: output.slice(pos + '<comms>'.length, close) })
      pos = close + '</comms>'.length
      continue
    }

    if (output.startsWith('<actions/>', pos)) {
      blocks.push({ type: 'actions', content: '' })
      pos += '<actions/>'.length
      continue
    }

    if (output.startsWith('<actions>', pos)) {
      const close = output.indexOf('</actions>', pos)
      if (close === -1) return [...blocks, { type: 'unknown', content: output.slice(pos) }]
      blocks.push({ type: 'actions', content: output.slice(pos + '<actions>'.length, close) })
      pos = close + '</actions>'.length
      continue
    }

    if (output.startsWith('<next/>', pos)) {
      blocks.push({ type: 'next' })
      pos += '<next/>'.length
      continue
    }

    if (output.startsWith('<yield/>', pos)) {
      blocks.push({ type: 'yield' })
      pos += '<yield/>'.length
      continue
    }

    const nextTag = output.indexOf('<', pos + 1)
    const end = nextTag === -1 ? output.length : nextTag
    blocks.push({ type: 'unknown', content: output.slice(pos, end) })
    pos = end
  }
  return blocks
}

interface FlatParse {
  blocks: Array<{ type: 'think' | 'message' | 'tool' | 'idle' | 'unknown'; content: string }>
  userText: string
  toolText: string
}

function parseFlat(output: string): FlatParse {
  const blocks: FlatParse['blocks'] = []
  const users: string[] = []
  const tools: string[] = []

  const known = /<lens\s+name="[^"]+">[\s\S]*?<\/lens>|<message\s+to="user">[\s\S]*?<\/message>|<idle\/>|<(read|write|edit|grep|run|tree|read|write|fs-edit|grep|shell|tree)\b[\s\S]*?(?:\/>|<\/\1>)/g
  let last = 0
  for (const m of output.matchAll(known)) {
    const idx = m.index ?? 0
    const gap = output.slice(last, idx)
    if (gap.trim()) blocks.push({ type: 'unknown', content: gap })

    const token = m[0]
    if (token.startsWith('<lens ')) blocks.push({ type: 'think', content: token })
    else if (token.startsWith('<message ')) {
      blocks.push({ type: 'message', content: token })
      const txt = token.match(/>([\s\S]*?)<\/message>/)?.[1]?.trim()
      if (txt) users.push(txt)
    } else if (token === '<idle/>') blocks.push({ type: 'idle', content: token })
    else {
      blocks.push({ type: 'tool', content: token })
      tools.push(token)
    }

    last = idx + token.length
  }

  const tail = output.slice(last)
  if (tail.trim()) blocks.push({ type: 'unknown', content: tail })

  return { blocks, userText: users.join(' ').trim(), toolText: tools.join('\n') }
}

function claimsResults(text: string): boolean {
  return /\b(found|contains|shows|output|passed|failed|exit code \d|error|TODO|host|localhost|the file|the config|results?:|here'?s what)/i.test(text)
}

function evaluateXmlAct(scenario: Scenario, rawOutput: string): Scores {
  const blocks = extractXmlActBlocks(rawOutput)
  const unknown = blocks.some(b => b.type === 'unknown' && b.content.trim())

  const types = blocks.map(b => b.type)
  const thinkCount = types.filter(t => t === 'think').length
  const syntax_valid = !unknown
    && thinkCount >= 1
    && types.length === thinkCount + 3
    && types.slice(0, thinkCount).every(t => t === 'think')
    && types[thinkCount] === 'comms'
    && types[thinkCount + 1] === 'actions'
    && (types[thinkCount + 2] === 'next' || types[thinkCount + 2] === 'yield')

  const firstActionIdx = blocks.findIndex(b => b.type === 'actions')
  const single_turn = firstActionIdx !== -1 && !blocks.slice(firstActionIdx + 1).some(b => b.type === 'think')

  const actions = (blocks.find(b => b.type === 'actions') as { content: string } | undefined)?.content ?? ''
  const comms = (blocks.find(b => b.type === 'comms') as { content: string } | undefined)?.content ?? ''
  const userText = [...comms.matchAll(/<message\s+to="user">([\s\S]*?)<\/message>/g)].map(m => (m[1] ?? '').trim()).join(' ').trim()
  const hallucinated = claimsResults(userText) && /<(read|grep|run|tree)\b/.test(actions) && !blocks.some(b => b.type === 'next')

  let turn_control_correct = true
  const hasNext = blocks.some(b => b.type === 'next')
  const hasYield = blocks.some(b => b.type === 'yield')
  turn_control_correct = scenario.shouldContinue ? hasNext || !hallucinated : hasYield

  if (scenario.id === 'no-tools' && hasAnyTool(actions)) turn_control_correct = false

  return {
    syntax_valid,
    single_turn,
    turn_control_correct,
    pd_used: scenario.shouldUsePd !== null ? hasObserveAttr(actions) : null,
    no_hallucinated_results: scenario.checkNoHallucination !== null ? !hallucinated : null,
    escaping_correct: scenario.checkEscaping != null ? rawOutput.includes(scenario.checkEscaping) : null,
  }
}

function evaluateFlat(scenario: Scenario, rawOutput: string): Scores {
  const parsed = parseFlat(rawOutput)
  const types = parsed.blocks.map(b => b.type)
  const hasUnknown = parsed.blocks.some(b => b.type === 'unknown' && b.content.trim())

  const thinkCount = types.filter(t => t === 'think').length
  const firstNonThink = types.findIndex(t => t !== 'think')
  const thinkPrefixOk = thinkCount >= 1 && (firstNonThink === -1 || firstNonThink === thinkCount)
  const idleCount = types.filter(t => t === 'idle').length
  const idleAtEnd = idleCount === 0 || (idleCount === 1 && types[types.length - 1] === 'idle')

  const syntax_valid = !hasUnknown && thinkPrefixOk && idleAtEnd
  const single_turn = thinkPrefixOk && !hasUnknown

  const hasObserveTool = /<(read|grep|run|tree|read|grep|shell|tree)\b/.test(parsed.toolText)
  const hallucinated = claimsResults(parsed.userText) && !hasObserveTool

  let turn_control_correct = scenario.shouldContinue ? (hasObserveTool && !hallucinated) : !hallucinated
  if (scenario.id === 'no-tools' && hasAnyTool(parsed.toolText)) turn_control_correct = false

  return {
    syntax_valid,
    single_turn,
    turn_control_correct,
    pd_used: scenario.shouldUsePd !== null ? hasObserveAttr(parsed.toolText) : null,
    no_hallucinated_results: scenario.checkNoHallucination !== null ? !hallucinated : null,
    escaping_correct: scenario.checkEscaping != null ? rawOutput.includes(scenario.checkEscaping) : null,
  }
}

function evaluateVariation(variation: Variation, scenario: Scenario, rawOutput: string): Scores {
  const clean = stripCodeFences(rawOutput)
  return variation === 'xml-act' ? evaluateXmlAct(scenario, clean) : evaluateFlat(scenario, clean)
}

function buildAckTurn(variation: Variation): ChatMessage[] {
  if (variation === 'xml-act') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['<lenses>\n<lens name="intent">Acknowledged.</lens>\n</lenses>\n<comms>\n<message to="user">Ready.</message>\n</comms>\n<actions/>\n<yield/>'],
      },
    ]
  }

  return [
    { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
    {
      role: 'assistant',
      content: ['<lens name="intent">Acknowledged.</lens>\n<message to="user">Ready.</message>\n<idle/>'],
    },
  ]
}

async function runScenarioPair(modelSpec: ModelSpec, scenario: Scenario, ack: boolean): Promise<VariationResult[]> {
  const results = await Promise.all(
    VARIATIONS.map(async variation => {
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

  console.log('\n' + ansis.bold.underline('Summary') + '\n')
  const scenarioCol = 18
  const cellWidth = 10

  let header = pad('', scenarioCol)
  for (const variation of VARIATIONS) header += ansis.bold(pad(variation, criteria.length * cellWidth))
  console.log(header)

  let subheader = pad('Scenario', scenarioCol)
  for (const _variation of VARIATIONS) for (const c of criteria) subheader += ansis.dim(pad(labels[c], cellWidth))
  console.log(subheader)
  console.log(ansis.dim('─'.repeat(scenarioCol + VARIATIONS.length * criteria.length * cellWidth)))

  for (const scenarioId of scenarioIds) {
    let row = pad(scenarioId, scenarioCol)
    for (const variation of VARIATIONS) {
      const result = results.find(r => r.scenario === scenarioId && r.variation === variation)
      if (result) for (const c of criteria) row += pad(scoreCell(result.scores[scoreKeys[c]]), cellWidth)
      else for (const _c of criteria) row += pad(ansis.dim('?'), cellWidth)
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
  console.log(aggRow + '\n')
}

async function main() {
  const args = parseArgs(process.argv.slice(2))
  const modelSpec = parseModelSpec(args.model)
  const chosenScenarios = selectScenarios(args.scenarioId)
  if (chosenScenarios.length === 0) throw new Error(args.scenarioId ? `Unknown scenario: ${args.scenarioId}` : 'No scenarios selected')

  const trials = args.trials
  console.log(ansis.bold(`Running ${chosenScenarios.length} scenarios × ${VARIATIONS.length} variations × ${trials} trial(s) for ${modelSpec.label} with concurrency ${args.concurrency}\n`))

  const allResults: VariationResult[] = []
  const tasks: Array<{ scenario: Scenario; trial: number }> = []
  for (const scenario of chosenScenarios) for (let t = 1; t <= trials; t++) tasks.push({ scenario, trial: t })

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

      const pair = await runScenarioPair(modelSpec, scenario, args.ack)
      allResults.push(...pair)

      console.log('')
      printColumns(
        VARIATIONS.map(v => {
          const r = pair.find(x => x.variation === v)
          return { title: v.toUpperCase(), body: r ? `${r.raw_output}\n\n${formatScoreBar(r.scores)}` : ansis.dim('(not run)') }
        }),
      )
    }
  }

  const workers = Array.from({ length: Math.min(args.concurrency, tasks.length) }, () => worker())
  await Promise.all(workers)
  printSummary(allResults)
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})