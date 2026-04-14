#!/usr/bin/env bun

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import path from 'node:path'
import type { AppEvent } from '@magnitudedev/agent'
import {
  MemoryProjection,
  TurnProjection,
  DisplayProjection,
  CompactionProjection,
  SessionContextProjection,
  ReplayProjection,
  AgentRoutingProjection,
  AgentStatusProjection,
} from '@magnitudedev/agent'

import { Agent } from '@magnitudedev/event-core'

type AnyRecord = Record<string, any>

const SESSIONS_ROOT = path.join(homedir(), '.magnitude', 'sessions')

function usage() {
  console.log(`Usage:
  session-inspect list
  session-inspect events <session-id> [--type <t1,t2>] [--from <N>] [--to <N>]
  session-inspect event <session-id> <index>
  session-inspect search <keyword> [<session-id> | --last <N>]
  session-inspect projection <session-id> <projection-name-or-all> [--at <index>]

Subcommands:
  list
  events
  event
  search
  projection`)
}

function replacer(_key: string, value: any): any {
  if (value instanceof Map) {
    return { __map__: true, entries: Array.from(value.entries()) }
  }
  if (value instanceof Set) {
    return Array.from(value.values())
  }
  return value
}

function expandHome(p: string): string {
  if (p.startsWith('~/')) return path.join(homedir(), p.slice(2))
  if (p === '~') return homedir()
  return p
}

function resolveSessionPath(id: string): string {
  const raw = expandHome(id)
  if (path.isAbsolute(raw) || raw.includes(path.sep)) return raw
  return path.join(SESSIONS_ROOT, id)
}

function loadEvents(sessionPath: string): any[] {
  const eventsPath = path.join(sessionPath, 'events.jsonl')
  if (!existsSync(eventsPath)) return []
  const text = readFileSync(eventsPath, 'utf8')
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, idx) => {
      try {
        return JSON.parse(line)
      } catch (err) {
        throw new Error(`Failed to parse events.jsonl line ${idx + 1} in ${sessionPath}: ${(err as Error).message}`)
      }
    })
}


function readMeta(sessionPath: string): AnyRecord {
  const metaPath = path.join(sessionPath, 'meta.json')
  if (!existsSync(metaPath)) return {}
  try {
    return JSON.parse(readFileSync(metaPath, 'utf8'))
  } catch {
    return {}
  }
}

function formatDate(value: any): string {
  if (value == null) return ''
  const d = new Date(value)
  if (Number.isNaN(d.getTime())) return String(value)
  return d.toISOString()
}

function formatTimestamp(value: any): string {
  if (value == null) return ''
  const d = new Date(value)
  if (!Number.isNaN(d.getTime())) return d.toISOString()
  return String(value)
}

function printTable(headers: string[], rows: string[][]) {
  const widths = headers.map((h, i) => {
    const maxCell = rows.reduce((m, r) => Math.max(m, (r[i] ?? '').length), 0)
    return Math.max(h.length, maxCell)
  })

  const fmt = (row: string[]) =>
    row
      .map((cell, i) => (cell ?? '').padEnd(widths[i], ' '))
      .join('  ')
      .trimEnd()

  console.log(fmt(headers))
  console.log(widths.map((w) => '-'.repeat(w)).join('  '))
  for (const r of rows) console.log(fmt(r))
}

function getSessionDirs(): string[] {
  if (!existsSync(SESSIONS_ROOT)) return []
  return readdirSync(SESSIONS_ROOT)
    .map((name) => path.join(SESSIONS_ROOT, name))
    .filter((full) => {
      try {
        return statSync(full).isDirectory()
      } catch {
        return false
      }
    })
}

function cmdList(args: string[]) {
  const limitArgIdx = args.indexOf('--limit')
  let limit = 20
  if (limitArgIdx !== -1 && args[limitArgIdx + 1]) {
    const n = Number.parseInt(args[limitArgIdx + 1], 10)
    if (Number.isFinite(n) && n > 0) limit = n
  }

  const sessions = getSessionDirs()
    .map((sessionPath) => {
      const id = path.basename(sessionPath)
      const meta = readMeta(sessionPath)
      const created = meta.created ?? meta.createdAt ?? meta.date ?? ''
      const title = meta.chatName
      const messages = meta.messageCount ?? ''
      return {
        id,
        title: String(title),
        createdRaw: created,
        created: formatDate(created),
        messages: String(messages)
      }
    })
    .sort((a, b) => {
      const ta = new Date(a.createdRaw).getTime()
      const tb = new Date(b.createdRaw).getTime()
      return (Number.isNaN(tb) ? 0 : tb) - (Number.isNaN(ta) ? 0 : ta)
    })
    .slice(0, limit)

  const rows = sessions.map((s) => [s.id, s.title, s.created, s.messages])
  printTable(['ID', 'Title', 'Date', 'Messages'], rows)
}

function parseOption(args: string[], name: string): string | undefined {
  const idx = args.indexOf(name)
  if (idx === -1) return undefined
  return args[idx + 1]
}

function cmdEvents(args: string[]) {
  const sessionId = args[0]
  if (!sessionId) {
    console.error('Missing <session-id>')
    usage()
    process.exit(1)
  }

  const typesRaw = parseOption(args, '--type')
  const fromRaw = parseOption(args, '--from')
  const toRaw = parseOption(args, '--to')

  const typeSet = typesRaw
    ? new Set(typesRaw.split(',').map((s) => s.trim()).filter(Boolean))
    : undefined

  const from = fromRaw != null ? Math.max(0, Number.parseInt(fromRaw, 10) || 0) : 0
  const to = toRaw != null ? Number.parseInt(toRaw, 10) : undefined

  const sessionPath = resolveSessionPath(sessionId)
  const events = loadEvents(sessionPath)

  const rows: string[][] = []
  for (let i = 0; i < events.length; i++) {
    if (i < from) continue
    if (to != null && i > to) break
    const e = events[i] ?? {}
    const t = e?.type ?? ''
    if (typeSet && !typeSet.has(String(t))) continue
    rows.push([String(i), String(t), formatTimestamp(e?.timestamp)])
  }

  printTable(['Index', 'Type', 'Timestamp'], rows)
}

function cmdEvent(args: string[]) {
  const sessionId = args[0]
  const indexRaw = args[1]
  if (!sessionId || indexRaw == null) {
    console.error('Usage: event <session-id> <index>')
    process.exit(1)
  }

  const index = Number.parseInt(indexRaw, 10)
  if (!Number.isFinite(index) || index < 0) {
    console.error('Index must be a non-negative integer')
    process.exit(1)
  }

  const sessionPath = resolveSessionPath(sessionId)
  const events = loadEvents(sessionPath)
  if (index >= events.length) {
    console.error(`Index out of range. Event count: ${events.length}`)
    process.exit(1)
  }

  console.log(JSON.stringify(events[index], replacer, 2))
}

function snippetAround(text: string, keyword: string, radius = 60): string {
  const lower = text.toLowerCase()
  const k = keyword.toLowerCase()
  const idx = lower.indexOf(k)
  if (idx === -1) return ''
  const start = Math.max(0, idx - radius)
  const end = Math.min(text.length, idx + k.length + radius)
  const prefix = start > 0 ? '...' : ''
  const suffix = end < text.length ? '...' : ''
  return `${prefix}${text.slice(start, end)}${suffix}`
}

function getLastNSessionIds(n: number): string[] {
  return getSessionDirs()
    .map((sessionPath) => {
      const id = path.basename(sessionPath)
      const meta = readMeta(sessionPath)
      const created = meta.created ?? meta.createdAt ?? meta.date ?? ''
      return { id, created }
    })
    .sort((a, b) => {
      const ta = new Date(a.created).getTime()
      const tb = new Date(b.created).getTime()
      return (Number.isNaN(tb) ? 0 : tb) - (Number.isNaN(ta) ? 0 : ta)
    })
    .slice(0, n)
    .map((x) => x.id)
}

function cmdSearch(args: string[]) {
  const keyword = args[0]
  if (!keyword) {
    console.error('Usage: search <keyword> [<session-id> | --last <N>]')
    process.exit(1)
  }

  let sessionIds: string[] = []
  if (args[1] === '--last') {
    const nRaw = args[2]
    const n = Math.max(1, Number.parseInt(nRaw ?? '1', 10) || 1)
    sessionIds = getLastNSessionIds(n)
  } else if (args[1]) {
    sessionIds = [args[1]]
  } else {
    sessionIds = getLastNSessionIds(1)
  }

  const rows: string[][] = []
  for (const id of sessionIds) {
    const sessionPath = resolveSessionPath(id)
    const events = loadEvents(sessionPath)
    for (let i = 0; i < events.length; i++) {
      const event = events[i]
      const text = JSON.stringify(event, replacer)
      if (!text) continue
      if (!text.toLowerCase().includes(keyword.toLowerCase())) continue
      rows.push([id, String(i), String(event?.type ?? ''), snippetAround(text, keyword)])
    }
  }

  printTable(['Session ID', 'Index', 'Type', 'Snippet'], rows)
}

const InspectAgent = Agent.define<AppEvent>()({
  name: 'SessionInspect',
  projections: [
    SessionContextProjection,
    CompactionProjection,
    TurnProjection,
    MemoryProjection,
    DisplayProjection,
    ReplayProjection,
    AgentRoutingProjection,
    AgentStatusProjection,
  ],
  workers: [],
  expose: {
    state: {
      sessionContext: SessionContextProjection,
      compaction: CompactionProjection,
      turn: TurnProjection,
      memory: MemoryProjection,
      display: DisplayProjection,
      replay: ReplayProjection,
      agentRouting: AgentRoutingProjection,
      agentStatus: AgentStatusProjection,
    }
  }
})

async function replayProjections(events: any[]): Promise<Record<string, any>> {
  const client = await InspectAgent.createClient()
  try {
    for (const ev of events) {
      await client.send(ev as AppEvent)
    }

    const result: Record<string, any> = {}

    // Global projections expose .get() -> state directly
    const globalKeys = ['sessionContext', 'agentRouting', 'agentStatus'] as const
    for (const key of globalKeys) {
      result[key] = await (client.state[key] as any).get()
    }

    // Forked projections expose .getFork(forkId) -> per-fork state
    const forkIds: (string | null)[] = [null]

    const forkedKeys = ['compaction', 'turn', 'memory', 'display', 'replay'] as const
    for (const key of forkedKeys) {
      const inst = client.state[key] as any
      const forks: Record<string, any> = {}
      for (const forkId of forkIds) {
        const forkState = await inst.getFork(forkId)
        forks[forkId === null ? 'root' : forkId] = forkState
      }
      result[key] = forks
    }

    return result
  } finally {
    await client.dispose()
  }
}

async function cmdProjection(args: string[]) {
  const sessionId = args[0]
  const projectionName = args[1]
  if (!sessionId || !projectionName) {
    console.error('Usage: projection <session-id> <projection-name-or-all> [--at <index>]')
    process.exit(1)
  }

  const atRaw = parseOption(args, '--at')
  const at = atRaw != null ? Number.parseInt(atRaw, 10) : undefined

  const sessionPath = resolveSessionPath(sessionId)
  let events = loadEvents(sessionPath)
  if (at != null && Number.isFinite(at) && at >= 0) {
    events = events.slice(0, at + 1)
  }

  const allStates = await replayProjections(events)

  if (projectionName.toLowerCase() === 'all') {
    console.log(JSON.stringify(allStates, replacer, 2))
    return
  }

  const projectionKeyMap: Record<string, string> = {
    memory: 'memory',
    turn: 'turn',
    display: 'display',
    compaction: 'compaction',
    sessioncontext: 'sessionContext',

    replay: 'replay',
    agentrouting: 'agentRouting',
    agentstatus: 'agentStatus',
  }

  const key = projectionKeyMap[projectionName.toLowerCase()]
  if (!key || !(key in allStates)) {
    console.error(`Unknown projection: ${projectionName}`)
    console.error('Supported: Memory, Turn, Display, Compaction, SessionContext, Replay, AgentRouting, AgentStatus, all')
    process.exit(1)
  }

  console.log(JSON.stringify(allStates[key], replacer, 2))
}

async function main() {
  const sub = process.argv[2]
  const args = process.argv.slice(3)

  try {
    switch (sub) {
      case 'list':
        cmdList(args)
        return
      case 'events':
        cmdEvents(args)
        return
      case 'event':
        cmdEvent(args)
        return
      case 'search':
        cmdSearch(args)
        return
      case 'projection':
        await cmdProjection(args)
        return
      default:
        usage()
        process.exit(sub ? 1 : 0)
    }
  } catch (err) {
    console.error((err as Error).message)
    process.exit(1)
  }
}

void main()