import type { ThinkingMessage, DisplayMessage } from '../types'
import { HIDE_THINKING_LABELS, TRAIT_LABELS } from '../constants'

/** Module-level map for held buffer state during streaming — keyed by thinking message ID.
 *  Used to avoid flicker when labels are split across chunk boundaries.
 */
export const heldBuffers = new Map<string, string>()

export function processThinkingChunk(
  step: ThinkingMessage,
  newText: string
): { contentToAppend: string; shouldSuppress: boolean } {
  if (!HIDE_THINKING_LABELS) {
    return { contentToAppend: newText, shouldSuppress: false }
  }

  const raw = (heldBuffers.get(step.id) ?? '') + newText

  // Only check [SKIP] suppression if step has no visible content yet
  if (step.content === '' && raw.includes('[SKIP]')) {
    heldBuffers.delete(step.id)
    return { contentToAppend: '', shouldSuppress: true }
  }

  // Strip known labels (+ optional trailing whitespace) from raw
  // Skip [SKIP] if step already has content — treat as ordinary text then
  let cleaned = raw
  for (const label of TRAIT_LABELS) {
    if (label === '[SKIP]' && step.content !== '') continue
    cleaned = cleaned.replaceAll(label + ' ', '')
    cleaned = cleaned.replaceAll(label + '\n', '')
    cleaned = cleaned.replaceAll(label + '\t', '')
    cleaned = cleaned.replaceAll(label, '')
  }

  // Check tail for potential label prefix
  const lastBracket = cleaned.lastIndexOf('[')
  if (lastBracket === -1) {
    heldBuffers.delete(step.id)
    return { contentToAppend: cleaned, shouldSuppress: false }
  }

  const suffix = cleaned.slice(lastBracket)
  const isPrefix = TRAIT_LABELS.some(l => l.startsWith(suffix))

  if (!isPrefix) {
    heldBuffers.delete(step.id)
    return { contentToAppend: cleaned, shouldSuppress: false }
  }

  heldBuffers.set(step.id, suffix)
  return { contentToAppend: cleaned.slice(0, lastBracket), shouldSuppress: false }
}

export function flushHeld(stepId: string): string {
  const held = heldBuffers.get(stepId)
  if (held !== undefined) {
    heldBuffers.delete(stepId)
    return held
  }
  return ''
}
