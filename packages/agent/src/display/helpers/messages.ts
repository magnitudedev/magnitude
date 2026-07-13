import { Option } from 'effect'
import type { DisplayMessage, ErrorDisplayMessage } from '../types'
import { present } from '../../errors'
import type { TurnOutcome } from '../../events'

export function toErrorDisplayMessage(id: string, outcome: TurnOutcome, timestamp: number): ErrorDisplayMessage | null {
  const p = present(outcome)
  if (p.surface !== 'inline') return null
  return {
    id,
    type: 'error',
    message: p.message,
    timestamp,
    cta: p.cta ? Option.some(p.cta) : Option.none(),
  }
}

export function toPreview(text: string): string {
  const normalized = text.replace(/\s+/g, ' ').trim()
  if (normalized.length <= 120) return normalized
  return normalized.slice(0, 117) + '...'
}
