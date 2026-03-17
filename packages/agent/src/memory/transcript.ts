import { readFile } from 'fs/promises'
import type { AppEvent, ResponsePart } from '../events'

const DEFAULT_TRANSCRIPT_CHAR_BUDGET = 35_000

function summarizeResultValue(value: unknown): string {
  const RESULT_SLICE = 300

  let text: string
  try {
    text = typeof value === 'string' ? value : JSON.stringify(value)
  } catch {
    text = String(value)
  }

  if (text.length <= RESULT_SLICE * 2) return text
  const head = text.slice(0, RESULT_SLICE)
  const tail = text.slice(-RESULT_SLICE)
  return `${head}\n...<truncated ${text.length - RESULT_SLICE * 2} chars>...\n${tail}`
}

function extractUserMessagesFromXml(xml: string): string[] {
  const matches: string[] = []
  const regex = /<message\b[^>]*\bto\s*=\s*(['"])user\1[^>]*>([\s\S]*?)<\/message>/gi
  let match: RegExpExecArray | null = regex.exec(xml)

  while (match) {
    const message = (match[2] ?? '').trim()
    if (message) matches.push(message)
    match = regex.exec(xml)
  }

  return matches
}

function extractUserMessages(parts: readonly ResponsePart[]): string {
  const messages: string[] = []

  for (const part of parts) {
    if (part.type !== 'text') continue
    messages.push(...extractUserMessagesFromXml(part.content))
  }

  return messages.join('\n')
}

function getEventTimestamp(event: AppEvent): string {
  const anyEvent = event as { timestamp?: string }
  return anyEvent.timestamp ?? 'unknown-time'
}

function toLine(index: number, event: AppEvent): string | null {
  const ts = getEventTimestamp(event)
  switch (event.type) {
    case 'oneshot_task': {
      return `[${index}] ${ts} oneshot_task\n${event.prompt}`
    }

    case 'user_message': {
      const text = event.content
        .filter((c: any) => c.type === 'text')
        .map((c: any) => c.text ?? '')
        .join('\n')
        .trim()
      if (!text) return null
      return `[${index}] ${ts} user_message\n${text}`
    }

    case 'turn_completed': {
      const text = extractUserMessages(event.responseParts)
      if (!text) return null
      return `[${index}] ${ts} assistant_turn_completed\n${text}`
    }

    case 'turn_unexpected_error':
      return `[${index}] ${ts} turn_unexpected_error\n${event.message}`

    case 'agent_created':
      return `[${index}] ${ts} agent_created role=${event.role} taskId=${event.taskId}`

    case 'agent_dismissed':
      return `[${index}] ${ts} agent_dismissed reason=${event.reason} result=${summarizeResultValue(event.result)}`

    default:
      return null
  }
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
  for (let i = 0; i < events.length; i++) {
    const line = toLine(i, events[i]!)
    if (line) lines.push(line)
  }
  return truncateTranscript(lines, opts?.maxChars ?? DEFAULT_TRANSCRIPT_CHAR_BUDGET)
}