import type { PasteApplyResult, PasteIntent } from './types'

export interface ResolvePasteIntentArgs {
  eventText?: string
  readClipboardText: () => string | null
  tryAddClipboardImage: () => Promise<boolean>
  tryAddImageFromFilePath: (rawPasteText: string) => Promise<boolean>
  inlinePastePillCharLimit: number
  blocked?: boolean
}

export async function resolvePasteIntent(args: ResolvePasteIntentArgs): Promise<PasteIntent> {
  if (args.blocked) return { kind: 'noop', reason: 'blocked' }

  const eventText = args.eventText ?? ''
  const pasteText = eventText.length > 0 ? eventText : args.readClipboardText()

  if (!pasteText) {
    const wasClipboardImage = await args.tryAddClipboardImage()
    if (wasClipboardImage) return { kind: 'add-clipboard-image' }
    return { kind: 'noop', reason: 'empty' }
  }

  const wasImagePath = await args.tryAddImageFromFilePath(pasteText)
  if (wasImagePath) return { kind: 'add-path-image', rawPath: pasteText }

  if (pasteText.length > args.inlinePastePillCharLimit) {
    return { kind: 'insert-segment-text', text: pasteText }
  }

  return { kind: 'insert-inline-text', text: pasteText }
}

export function resolvePasteOutcomeFromApplyResult(result: PasteApplyResult): boolean {
  return result.kind !== 'noop'
}
