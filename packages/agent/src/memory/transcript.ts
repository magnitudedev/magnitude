import { readFile } from 'fs/promises'
import type { AppEvent } from '../events'

const DEFAULT_TRANSCRIPT_CHAR_BUDGET = 35_000

function getEventTimestamp(event: AppEvent): string {
  const anyEvent = event as { timestamp?: string }
  return anyEvent.timestamp ?? 'unknown-time'
}

function toLine(index: number, event: AppEvent): string | null {
  const ts = getEventTimestamp(event)
  switch (event.type) {
    case 'oneshot_task':
      return `[${index}] ${ts} oneshot_task\n${event.prompt}`

    case 'user_message': {
      const contentText = event.content
        .filter((c): c is Extract<(typeof event.content)[number], { type: 'text' }> => c.type === 'text')
        .map(c => c.text)
        .join('\n')
        .trim()
      if (!contentText) return null
      return `[${index}] ${ts} user_message\n${contentText}`
    }

    case 'turn_outcome':
      if (event.outcome._tag !== 'UnexpectedError') return null
      return `[${index}] ${ts} turn_outcome.unexpected_error\n${event.outcome.message}`

    case 'agent_created':
      return `[${index}] ${ts} agent_created role=${event.role} taskId=${event.taskId}`

    default:
      return null
  }
}

function assistantTurnLine(index: number, timestamp: string, text: string): string | null {
  if (!text) return null
  return `[${index}] ${timestamp} assistant_turn_outcome\n${text}`
}

function truncateTranscript(lines: string[], maxChars: number): string {
  if (lines.length === 0) return ''
  const sep = '\n\n'
  let total = lines.reduce((sum, l) => sum + l.length, 0) + sep.length * (lines.length - 1)
  let start = 0
  while (start < lines.length && total > maxChars) {
    total -= lines[start]!.length + (start < lines.length - 1 ? sep.length : 0)
    start++
  }
  return lines.slice(start).join(sep)
}

export async function readEventsJsonl(eventsPath: string): Promise<AppEvent[]> {
  const raw = await readFile(eventsPath, 'utf8')
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line) as AppEvent)
}

export function buildExtractionTranscript(
  events: AppEvent[],
  opts?: { maxChars?: number }
): string {
  const lines: string[] = []
  const pendingUserMessages = new Map<string, { startIndex: number; parts: string[] }>()

  for (let i = 0; i < events.length; i++) {
    const event = events[i]!

    if (event.type === 'message_start' && event.destination.kind === 'user') {
      pendingUserMessages.set(event.id, { startIndex: i, parts: [] })
      continue
    }

    if (event.type === 'message_chunk') {
      const pending = pendingUserMessages.get(event.id)
      if (pending) pending.parts.push(event.text)
      continue
    }

    if (event.type === 'message_end') {
      const pending = pendingUserMessages.get(event.id)
      if (pending) {
        pendingUserMessages.delete(event.id)
        const text = pending.parts.join('').trim()
        const line = assistantTurnLine(i, getEventTimestamp(event), text)
        if (line) lines.push(line)
      }
      continue
    }

    const line = toLine(i, event)
    if (line) lines.push(line)
  }

  return truncateTranscript(lines, opts?.maxChars ?? DEFAULT_TRANSCRIPT_CHAR_BUDGET)
}