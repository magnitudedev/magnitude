import type { PasteApplyResult, PasteEffectsResult } from './types'

export function derivePasteEffects(result: PasteApplyResult): PasteEffectsResult {
  switch (result.kind) {
    case 'inserted-inline-text':
    case 'inserted-segment-text':
      return {
        shouldReportInserted: true,
        shouldBumpBulkInsertEpoch: true,
        focusNudgeRequested: false,
      }
    case 'added-clipboard-image':
    case 'added-path-image':
      return {
        shouldReportInserted: true,
        shouldBumpBulkInsertEpoch: false,
        focusNudgeRequested: false,
      }
    case 'noop':
      return {
        shouldReportInserted: false,
        shouldBumpBulkInsertEpoch: false,
        focusNudgeRequested: false,
        feedbackReason: result.reason,
      }
  }
}
