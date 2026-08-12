import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'

const EXACT_PROMPT = 'use the GitHits CLI to inspect https://github.com/magnitudedev/browser-agent. Search its source and documentation for supported or recommended AI models. Do not use web_fetch, local file tools, githits pkg info, or inspect Magnitude’s local package.json. Report repository paths and source evidence.'
const EXPECTED_MODEL = 'claude-sonnet-4-6'

type Event = {
  readonly type: string
  readonly timestamp?: number
  readonly text?: string
  readonly toolCallId?: string
  readonly toolKey?: string
  readonly event?: {
    readonly _tag?: string
    readonly input?: { readonly command?: string }
  }
}

function fail(message: string): never {
  console.error(`FAIL: ${message}`)
  process.exit(1)
}

function notRun(message: string): never {
  console.error(`NOT RUN: ${message}`)
  process.exit(2)
}

const sessionFlag = process.argv.indexOf('--session')
const sessionId = sessionFlag >= 0 ? process.argv[sessionFlag + 1] : undefined
if (!sessionId) notRun('pass --session <id> for a completed authenticated live run')

const sessionDirectory = join(homedir(), '.magnitude', 'sessions', sessionId)
let events: Event[]
let projections: any
try {
  const [eventText, projectionText] = await Promise.all([
    readFile(join(sessionDirectory, 'events.jsonl'), 'utf8'),
    readFile(join(sessionDirectory, 'projections.json'), 'utf8'),
  ])
  events = eventText.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line) as Event)
  projections = JSON.parse(projectionText)
} catch (error) {
  notRun(`session ${sessionId} is unavailable or incomplete: ${String(error)}`)
}

const primary = projections?.projections?.AgentToolkit?.[0]?.[1]?.config?.bySlot?.primary?.config
const modelId = String(primary?.providerModelId ?? '')
if (!modelId.includes(EXPECTED_MODEL)) {
  notRun(`session model is ${modelId || 'unknown'}; expected ${EXPECTED_MODEL}`)
}

const promptEvent = events.find((event) => event.type === 'user_message' && event.text === EXACT_PROMPT)
if (!promptEvent?.timestamp) fail('exact browser-agent prompt is absent')
const afterPrompt = events.filter((event) => (event.timestamp ?? 0) >= promptEvent.timestamp!)
const started = afterPrompt.filter((event) => event.type === 'tool_event' && event.event?._tag === 'ToolExecutionStarted')

const skillIndex = started.findIndex((event) => event.toolKey === 'skill')
if (skillIndex < 0) fail('githits-code skill was not loaded')

const discoveryKeys = new Set(['shell', 'fileRead', 'fileTree', 'fileSearch', 'webFetch', 'webSearch'])
const firstDiscovery = started.slice(skillIndex + 1).find((event) => discoveryKeys.has(event.toolKey ?? ''))
if (!firstDiscovery) fail('no repository-discovery action followed skill loading')
if (firstDiscovery.toolKey !== 'shell' || !/^\s*(githits\b|npx\s+-y\s+githits@latest\b)/.test(firstDiscovery.event?.input?.command ?? '')) {
  fail(`first repository-discovery action was ${firstDiscovery.toolKey ?? 'unknown'}, not GitHits CLI`)
}

const endedCounts = new Map<string, number>()
for (const event of afterPrompt) {
  if (event.type === 'tool_event' && event.event?._tag === 'ToolExecutionEnded' && event.toolCallId) {
    endedCounts.set(event.toolCallId, (endedCounts.get(event.toolCallId) ?? 0) + 1)
  }
}
for (const event of started) {
  if (!event.toolCallId || endedCounts.get(event.toolCallId) !== 1) {
    fail(`tool ${event.toolCallId ?? 'unknown'} did not emit exactly one terminal event`)
  }
}

const finalText = afterPrompt.filter((event) => event.type === 'message_chunk').map((event) => event.text ?? '').join('')
if (!/browser-agent/i.test(finalText) || !/[\w.-]+\/[\w./-]+/.test(finalText)) {
  fail('final answer does not contain browser-agent repository paths/source evidence')
}

const elapsed = Math.max(...afterPrompt.map((event) => event.timestamp ?? promptEvent.timestamp)) - promptEvent.timestamp
if (elapsed > 180_000) fail(`run exceeded 180 seconds (${elapsed}ms)`)

console.log(`PASS: ${sessionId} routed through GitHits and completed in ${elapsed}ms`)
