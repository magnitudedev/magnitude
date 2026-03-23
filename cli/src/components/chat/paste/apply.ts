import { createId } from '@magnitudedev/generate-id'
import { applyTextEditWithPastesAndMentions, insertPasteSegment } from '../../../utils/strings'
import type { InputValue } from '../../../types/store'
import type { PasteApplyResult, PasteIntent } from './types'

export function applyPasteIntent(args: {
  intent: PasteIntent
  setInputValue: (updater: (prev: InputValue) => InputValue) => void
}): PasteApplyResult {
  const { intent } = args
  if (intent.kind === 'insert-inline-text') {
    args.setInputValue((prev) =>
      applyTextEditWithPastesAndMentions(prev, prev.cursorPosition, prev.cursorPosition, intent.text),
    )
    return { kind: 'inserted-inline-text' }
  }

  if (intent.kind === 'insert-segment-text') {
    args.setInputValue((prev) => insertPasteSegment(prev, intent.text, createId()))
    return { kind: 'inserted-segment-text' }
  }

  if (intent.kind === 'add-clipboard-image') {
    return { kind: 'added-clipboard-image' }
  }

  if (intent.kind === 'add-path-image') {
    return { kind: 'added-path-image' }
  }

  return { kind: 'noop', reason: intent.reason }
}
