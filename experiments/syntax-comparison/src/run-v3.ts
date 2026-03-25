import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { Effect, Layer, Stream } from 'effect'
import ansis from 'ansis'

import type { ChatMessage } from '@magnitudedev/llm-core'
import { ModelResolver, SimpleChat, createProviderClient, makeModelResolver, makeNoopTracer } from '@magnitudedev/providers'
import { scenarios } from './scenarios'
import type { ModelSpec, Scenario, Scores } from './types'

type Variation = 'xml-act' | 'observe-mutate' | 'token-delim' | 'bbcode' | 'parens' | 'curly'

interface VariationResult {
  model: string
  variation: Variation
  scenario: string
  raw_output: string
  scores: Scores
}

const VARIATIONS: Variation[] = ['xml-act', 'observe-mutate', /* 'token-delim', 'bbcode', 'parens', 'curly' */]

let cachedClientPromise: ReturnType<typeof createProviderClient> | null = null

async function getProviderClient() {
  if (!cachedClientPromise) cachedClientPromise = createProviderClient({ slots: ['primary'] as const })
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
  const file = variation === 'xml-act'
    ? 'xml-act.txt'
    : variation === 'observe-mutate'
      ? 'observe-mutate.txt'
      : variation === 'token-delim'
        ? 'token-delim.txt'
        : variation === 'bbcode'
          ? 'bbcode.txt'
          : variation === 'curly'
            ? 'curly.txt'
            : 'parens.txt'
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
  const criteriaKeys: { key: keyof Scores; label: string }[] = [
    { key: 'syntax_valid', label: 'syntax' },
    { key: 'single_turn', label: '1-turn' },
    { key: 'turn_control_correct', label: 'turn' },
    { key: 'pd_used', label: 'pd' },
    { key: 'no_hallucinated_results', label: 'halluc' },
    { key: 'escaping_correct', label: 'escape' },
  ]

  const totalWidth = process.stdout.columns || 120
  const colWidth = Math.floor((totalWidth - 12) / 5)

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

function printColumns(columns: { title: string; body: string }[]) {
  const totalWidth = process.stdout.columns || 180
  const numCols = columns.length
  const colWidth = Math.floor((totalWidth - (numCols - 1) * 3) / numCols)
  const sep = ansis.dim(' │ ')

  const allLines = columns.map(c => wrapText(c.body, colWidth))
  const maxLines = Math.max(...allLines.map(l => l.length))
  const colors = [ansis.yellow, ansis.magenta, ansis.cyan, ansis.green, ansis.blue]

  console.log(columns.map((c, i) => {
    const color = colors[i % colors.length]
    return color.bold(i < numCols - 1 ? pad(c.title, colWidth) : c.title)
  }).join(sep))

  console.log(columns.map(() => ansis.dim('─'.repeat(colWidth))).join(sep))

  for (let i = 0; i < maxLines; i++) {
    console.log(columns.map((_, ci) => {
      const line = allLines[ci][i] ?? ''
      return ci < numCols - 1 ? pad(line, colWidth) : line
    }).join(sep))
  }
}

// structural block parsing + validation

type Block =
  | { type: 'think'; name: string; content: string }
  | { type: 'observe'; content: string }
  | { type: 'mutate'; content: string }
  | { type: 'actions'; content: string }
  | { type: 'comms'; content: string }
  | { type: 'idle' }
  | { type: 'next' }
  | { type: 'yield' }
  | { type: 'unknown'; content: string }

function extractBlocks(variation: Variation, output: string): Block[] {
  return variation === 'xml-act'
    ? extractXmlActBlocks(output)
    : variation === 'observe-mutate'
      ? extractObserveMutateBlocks(output)
      : variation === 'token-delim'
        ? extractTokenDelimBlocks(output)
        : variation === 'bbcode'
          ? extractBbcodeBlocks(output)
          : variation === 'parens'
            ? extractParensBlocks(output)
            : extractCurlyBlocks(output)
}

function extractXmlActBlocks(output: string): Block[] {
  const blocks: Block[] = []
  let pos = 0

  while (pos < output.length) {
    while (pos < output.length && /\s/.test(output[pos])) pos++
    if (pos >= output.length) break

    if (output.startsWith('<lenses>', pos)) {
      const close = output.indexOf('</lenses>', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      const inner = output.slice(pos + '<lenses>'.length, close)
      const lensRe = /<lens\s+name="([^"]+)">([\s\S]*?)<\/lens>/g
      let m: RegExpExecArray | null = null
      while ((m = lensRe.exec(inner)) !== null) {
        blocks.push({ type: 'think', name: m[1], content: m[2] })
      }
      pos = close + '</lenses>'.length
      continue
    }

    if (output.startsWith('<comms>', pos)) {
      const close = output.indexOf('</comms>', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
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
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
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

function extractObserveMutateBlocks(output: string): Block[] {
  const blocks: Block[] = []
  let pos = 0

  while (pos < output.length) {
    while (pos < output.length && /\s/.test(output[pos])) pos++
    if (pos >= output.length) break

    if (output.startsWith('<reasoning', pos)) {
      const open = output.slice(pos).match(/^<reasoning\b([^>]*)>/)
      if (!open) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      const openTag = open[0]
      const name = /about="([^"]+)"/.exec(open[1] ?? '')?.[1] ?? ''
      const contentStart = pos + openTag.length
      const close = output.indexOf('</reasoning>', contentStart)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'think', name, content: output.slice(contentStart, close) })
      pos = close + '</reasoning>'.length
      continue
    }

    if (output.startsWith('<observe>', pos)) {
      const close = output.indexOf('</observe>', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'observe', content: output.slice(pos + '<observe>'.length, close) })
      pos = close + '</observe>'.length
      continue
    }

    if (output.startsWith('<mutate>', pos)) {
      const close = output.indexOf('</mutate>', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'mutate', content: output.slice(pos + '<mutate>'.length, close) })
      pos = close + '</mutate>'.length
      continue
    }

    if (output.startsWith('<idle/>', pos)) {
      blocks.push({ type: 'idle' })
      pos += '<idle/>'.length
      continue
    }

    const nextTag = output.indexOf('<', pos + 1)
    const end = nextTag === -1 ? output.length : nextTag
    blocks.push({ type: 'unknown', content: output.slice(pos, end) })
    pos = end
  }

  return blocks
}

function extractTokenDelimBlocks(output: string): Block[] {
  const blocks: Block[] = []
  let pos = 0

  while (pos < output.length) {
    while (pos < output.length && /\s/.test(output[pos])) pos++
    if (pos >= output.length) break

    if (output.startsWith('<|thk', pos)) {
      const open = output.slice(pos).match(/^<\|thk\b([\s\S]*?)\|\>/)
      if (!open) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      const openTag = open[0]
      const name = /about="([^"]+)"/.exec(open[1] ?? '')?.[1] ?? ''
      const contentStart = pos + openTag.length
      const close = output.indexOf('<|/thk|>', contentStart)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'think', name, content: output.slice(contentStart, close) })
      pos = close + '<|/thk|>'.length
      continue
    }

    if (output.startsWith('<|obs|>', pos)) {
      const close = output.indexOf('<|/obs|>', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'observe', content: output.slice(pos + '<|obs|>'.length, close) })
      pos = close + '<|/obs|>'.length
      continue
    }

    if (output.startsWith('<|mut|>', pos)) {
      const close = output.indexOf('<|/mut|>', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'mutate', content: output.slice(pos + '<|mut|>'.length, close) })
      pos = close + '<|/mut|>'.length
      continue
    }

    if (output.startsWith('<|idle|>', pos)) {
      blocks.push({ type: 'idle' })
      pos += '<|idle|>'.length
      continue
    }

    const nextTag = output.indexOf('<', pos + 1)
    const end = nextTag === -1 ? output.length : nextTag
    blocks.push({ type: 'unknown', content: output.slice(pos, end) })
    pos = end
  }

  return blocks
}

function extractBbcodeBlocks(output: string): Block[] {
  const blocks: Block[] = []
  let pos = 0

  while (pos < output.length) {
    while (pos < output.length && /\s/.test(output[pos])) pos++
    if (pos >= output.length) break

    if (output.startsWith('[thk', pos)) {
      const open = output.slice(pos).match(/^\[thk\b([\s\S]*?)\]/)
      if (!open) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      const openTag = open[0]
      const name = /about="([^"]+)"/.exec(open[1] ?? '')?.[1] ?? ''
      const contentStart = pos + openTag.length
      const close = output.indexOf('[/thk]', contentStart)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'think', name, content: output.slice(contentStart, close) })
      pos = close + '[/thk]'.length
      continue
    }

    if (output.startsWith('[obs]', pos)) {
      const close = output.indexOf('[/obs]', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'observe', content: output.slice(pos + '[obs]'.length, close) })
      pos = close + '[/obs]'.length
      continue
    }

    if (output.startsWith('[mut]', pos)) {
      const close = output.indexOf('[/mut]', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'mutate', content: output.slice(pos + '[mut]'.length, close) })
      pos = close + '[/mut]'.length
      continue
    }

    if (output.startsWith('[idle]', pos)) {
      blocks.push({ type: 'idle' })
      pos += '[idle]'.length
      continue
    }

    const nextTag = output.indexOf('[', pos + 1)
    const nextAngle = output.indexOf('<', pos + 1)
    const nextParen = output.indexOf('(', pos + 1)
    const candidates = [nextTag, nextAngle, nextParen].filter(x => x !== -1)
    const end = candidates.length === 0 ? output.length : Math.min(...candidates)
    blocks.push({ type: 'unknown', content: output.slice(pos, end) })
    pos = end
  }

  return blocks
}

function extractParensBlocks(output: string): Block[] {
  const blocks: Block[] = []
  let pos = 0

  while (pos < output.length) {
    while (pos < output.length && /\s/.test(output[pos])) pos++
    if (pos >= output.length) break

    if (output.startsWith('(reasoning', pos)) {
      const open = output.slice(pos).match(/^\(reasoning\b([^)]*)\)/)
      if (!open) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      const openTag = open[0]
      const name = /about="([^"]+)"/.exec(open[1] ?? '')?.[1] ?? ''
      const contentStart = pos + openTag.length
      const close = output.indexOf('(/reasoning)', contentStart)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'think', name, content: output.slice(contentStart, close) })
      pos = close + '(/reasoning)'.length
      continue
    }

    if (output.startsWith('(observe)', pos)) {
      const close = output.indexOf('(/observe)', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'observe', content: output.slice(pos + '(observe)'.length, close) })
      pos = close + '(/observe)'.length
      continue
    }

    if (output.startsWith('(mutate)', pos)) {
      const close = output.indexOf('(/mutate)', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'mutate', content: output.slice(pos + '(mutate)'.length, close) })
      pos = close + '(/mutate)'.length
      continue
    }

    if (output.startsWith('(idle/)', pos)) {
      blocks.push({ type: 'idle' })
      pos += '(idle/)'.length
      continue
    }

    const nextParen = output.indexOf('(', pos + 1)
    const nextAngle = output.indexOf('<', pos + 1)
    const nextBracket = output.indexOf('[', pos + 1)
    const candidates = [nextParen, nextAngle, nextBracket].filter(x => x !== -1)
    const end = candidates.length === 0 ? output.length : Math.min(...candidates)
    blocks.push({ type: 'unknown', content: output.slice(pos, end) })
    pos = end
  }

  return blocks
}

function extractCurlyBlocks(output: string): Block[] {
  const blocks: Block[] = []
  let pos = 0

  while (pos < output.length) {
    while (pos < output.length && /\s/.test(output[pos])) pos++
    if (pos >= output.length) break

    if (output.startsWith('{reasoning', pos)) {
      const open = output.slice(pos).match(/^\{reasoning\b([^}]*)\}/)
      if (!open) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      const openTag = open[0]
      const name = /about="([^"]+)"/.exec(open[1] ?? '')?.[1] ?? ''
      const contentStart = pos + openTag.length
      const close = output.indexOf('{/reasoning}', contentStart)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'think', name, content: output.slice(contentStart, close) })
      pos = close + '{/reasoning}'.length
      continue
    }

    if (output.startsWith('{observe}', pos)) {
      const close = output.indexOf('{/observe}', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'observe', content: output.slice(pos + '{observe}'.length, close) })
      pos = close + '{/observe}'.length
      continue
    }

    if (output.startsWith('{mutate}', pos)) {
      const close = output.indexOf('{/mutate}', pos)
      if (close === -1) {
        blocks.push({ type: 'unknown', content: output.slice(pos) })
        break
      }
      blocks.push({ type: 'mutate', content: output.slice(pos + '{mutate}'.length, close) })
      pos = close + '{/mutate}'.length
      continue
    }

    if (output.startsWith('{idle/}', pos)) {
      blocks.push({ type: 'idle' })
      pos += '{idle/}'.length
      continue
    }

    const nextCurly = output.indexOf('{', pos + 1)
    const nextAngle = output.indexOf('<', pos + 1)
    const candidates = [nextCurly, nextAngle].filter(x => x !== -1)
    const end = candidates.length === 0 ? output.length : Math.min(...candidates)
    blocks.push({ type: 'unknown', content: output.slice(pos, end) })
    pos = end
  }

  return blocks
}

function hasUnknownBlocks(blocks: Block[]): boolean {
  return blocks.some(b => b.type === 'unknown' && b.content.trim().length > 0)
}

function syntaxValid(variation: Variation, blocks: Block[]): boolean {
  if (hasUnknownBlocks(blocks)) return false

  const types = blocks.map(b => b.type)

  if (variation === 'xml-act') {
    const thinkCount = types.filter(t => t === 'think').length
    if (thinkCount < 1) return false
    if (types.length !== thinkCount + 3) return false
    if (!types.slice(0, thinkCount).every(t => t === 'think')) return false
    if (types[thinkCount] !== 'comms') return false
    if (types[thinkCount + 1] !== 'actions') return false
    return types[thinkCount + 2] === 'next' || types[thinkCount + 2] === 'yield'
  }

  const thinkCount = types.filter(t => t === 'think').length
  if (thinkCount < 1) return false
  if (!types.slice(0, thinkCount).every(t => t === 'think')) return false

  const action = types[thinkCount]
  if (action !== 'observe' && action !== 'mutate') return false

  if (action === 'observe') return types.length === thinkCount + 1
  if (types.length === thinkCount + 1) return true
  return types.length === thinkCount + 2 && types[thinkCount + 1] === 'idle'
}

function isSingleTurn(variation: Variation, blocks: Block[]): boolean {
  const firstActionIdx = blocks.findIndex(b => b.type === 'actions' || b.type === 'observe' || b.type === 'mutate')
  if (firstActionIdx === -1) return false
  const thinkAfterAction = blocks.slice(firstActionIdx + 1).some(b => b.type === 'think')
  if (thinkAfterAction) return false

  if (variation === 'xml-act') {
    return blocks.filter(b => b.type === 'actions').length === 1
  }
  return blocks.filter(b => b.type === 'observe' || b.type === 'mutate').length === 1
}

function continues(variation: Variation, blocks: Block[]): boolean {
  if (variation === 'xml-act') return blocks.some(b => b.type === 'next')
  const hasObserve = blocks.some(b => b.type === 'observe')
  const hasMutate = blocks.some(b => b.type === 'mutate')
  const hasIdle = blocks.some(b => b.type === 'idle')
  return hasObserve || (hasMutate && !hasIdle)
}

function getBlockContent(blocks: Block[], type: Block['type']): string {
  const b = blocks.find(x => x.type === type) as Extract<Block, { content: string }> | undefined
  return b?.content ?? ''
}

function hasAnyTool(content: string): boolean {
  return /<(read|write|edit|search|run|tree)\b|\((read|write|edit|search|run|tree)\b|\{(read|write|edit|search|run|tree)\b|\[(read|write|edit|search|run|tree)\b/.test(content)
}

function hasObserveAttr(content: string): boolean {
  return /\bobserve="[^"]*"/.test(content)
}

function usesPd(variation: Variation, blocks: Block[]): boolean {
  if (variation === 'xml-act') {
    const actions = getBlockContent(blocks, 'actions')
    return hasObserveAttr(actions)
  }
  return blocks.some(b => b.type === 'observe')
}

function extractUserFacingText(variation: Variation, blocks: Block[]): string {
  if (variation === 'xml-act') {
    const comms = getBlockContent(blocks, 'comms')
    const msgs = [...comms.matchAll(/<message\s+to="user">([\s\S]*?)<\/message>/g)].map(m => (m[1] ?? '').trim())
    return msgs.join(' ').trim()
  }

  const mutate = getBlockContent(blocks, 'mutate')
  if (variation === 'parens') {
    const msgs = [...mutate.matchAll(/\(message\s+to="user"\)([\s\S]*?)\(\/message\)/g)].map(m => (m[1] ?? '').trim())
    return msgs.join(' ').trim()
  }

  if (variation === 'curly') {
    const msgs = [...mutate.matchAll(/\{message\s+to="user"\}([\s\S]*?)\{\/message\}/g)].map(m => (m[1] ?? '').trim())
    return msgs.join(' ').trim()
  }

  const msgs = [...mutate.matchAll(/<message\s+to="user">([\s\S]*?)<\/message>/g)].map(m => (m[1] ?? '').trim())
  return msgs.join(' ').trim()
}

function hallucinatesResults(variation: Variation, blocks: Block[]): boolean {
  const userText = extractUserFacingText(variation, blocks)
  const claimsResults = /\b(found|contains|shows|output|passed|failed|exit code \d|error|TODO|host|localhost|the file|the config|results?:|here'?s what)/i.test(userText)

  if (variation === 'xml-act') {
    const actions = getBlockContent(blocks, 'actions')
    const hasReadOrSearchOrRun = /<(read|search|run)\b/.test(actions)
    return claimsResults && hasReadOrSearchOrRun && !continues(variation, blocks)
  }

  const hasObserve = blocks.some(b => b.type === 'observe')
  return claimsResults && !hasObserve
}

function stripCodeFences(output: string): string {
  let s = output.trim()
  if (s.startsWith('```')) {
    s = s.replace(/^```[^\n]*\n?/, '').replace(/\n?```\s*$/, '')
  }
  return s.trim()
}



function evaluateVariation(variation: Variation, scenario: Scenario, rawOutput: string): Scores {
  rawOutput = stripCodeFences(rawOutput)
  const blocks = extractBlocks(variation, rawOutput)

  const syntax_valid = syntaxValid(variation, blocks)
  const single_turn = isSingleTurn(variation, blocks)
  const hallucinated = hallucinatesResults(variation, blocks)

  let turn_control_correct = true
  if (variation === 'xml-act') {
    const endsWithNext = blocks.some(b => b.type === 'next')
    const endsWithYield = blocks.some(b => b.type === 'yield')
    turn_control_correct = scenario.shouldContinue ? endsWithNext : endsWithYield
    if (scenario.shouldContinue && !endsWithNext) {
      turn_control_correct = !hallucinated
    }
  } else {
    const hasObserve = blocks.some(b => b.type === 'observe')
    const hasMutate = blocks.some(b => b.type === 'mutate')
    const hasIdle = blocks.some(b => b.type === 'idle')

    if (scenario.shouldContinue) {
      turn_control_correct = hasObserve && !hallucinated
    } else {
      turn_control_correct = hasMutate && hasIdle && !hallucinated
    }
  }

  if (scenario.id === 'no-tools') {
    const actionContent = blocks
      .filter(b => b.type === 'actions' || b.type === 'observe' || b.type === 'mutate')
      .map(b => ('content' in b ? b.content : ''))
      .join('\n')
    if (hasAnyTool(actionContent)) {
      turn_control_correct = false
    }
  }

  return {
    syntax_valid,
    single_turn,
    turn_control_correct,
    pd_used: scenario.shouldUsePd !== null ? usesPd(variation, blocks) : null,
    no_hallucinated_results: scenario.checkNoHallucination !== null ? !hallucinated : null,
    escaping_correct: scenario.checkEscaping != null ? rawOutput.includes(scenario.checkEscaping) : null,
  }
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

  if (variation === 'observe-mutate') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['<reasoning about="intent">Demonstrating format.</reasoning>\n<observe>\n<read path="example.txt" observe=".content"/>\n</observe>'],
      },
    ]
  }

  if (variation === 'token-delim') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['<|thk about="intent"|>Demonstrating format.<|/thk|>\n<|obs|>\n<read path="example.txt" observe=".content"/>\n<|/obs|>'],
      },
    ]
  }

  if (variation === 'bbcode') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['[thk about="intent"]Demonstrating format.[/thk]\n[obs]\n<read path="example.txt" observe=".content"/>\n[/obs]'],
      },
    ]
  }

  if (variation === 'parens') {
    return [
      { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
      {
        role: 'assistant',
        content: ['(reasoning about="intent")Demonstrating format.(/reasoning)\n(observe)\n(read path="example.txt" observe=".content"/)\n(/observe)'],
      },
    ]
  }

  return [
    { role: 'user', content: ['Acknowledge to demonstrate understanding of response syntax'] },
    {
      role: 'assistant',
      content: ['{reasoning about="intent"}Demonstrating format.{/reasoning}\n{observe}\n{read path="example.txt" observe=".content"/}\n{/observe}'],
    },
  ]
}

async function runScenarioPair(modelSpec: ModelSpec, scenario: Scenario, ack: boolean): Promise<VariationResult[]> {
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
  console.log(ansis.bold(`Running ${chosenScenarios.length} scenarios × 5 variations × ${trials} trial(s) for ${modelSpec.label} with concurrency ${args.concurrency}\n`))

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

      const pair = await runScenarioPair(modelSpec, scenario, args.ack)
      allResults.push(...pair)

      console.log('')
      printColumns(
        VARIATIONS.map(v => {
          const r = pair.find(r => r.variation === v)
          return {
            title: v.toUpperCase(),
            body: r ? `${r.raw_output}\n\n${formatScoreBar(r.scores)}` : ansis.dim('(not run)'),
          }
        }),
      )

      printRunningStats(allResults)
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