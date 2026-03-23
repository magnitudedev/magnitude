export interface PasteEventLike {
  text?: string
}

export interface PasteIngestShortcut {
  type: 'shortcut'
}

export interface PasteIngestNative {
  type: 'native'
  text?: string
  replayedFromDeferred?: boolean
}

export type PasteIngestRequest = PasteIngestShortcut | PasteIngestNative

export type PasteIngestDropReason = 'disabled' | 'not-paste-shortcut' | 'native-duplicate-after-fallback-success'

export type PasteIngestOutcome =
  | { kind: 'fallback-requested' }
  | { kind: 'native-event'; text?: string; replayedFromDeferred: boolean }
  | { kind: 'dropped'; reason: PasteIngestDropReason }

export interface PendingPasteAttempt {
  id: number
  status: 'pending' | 'fallback-in-flight' | 'fallback-succeeded' | 'fallback-empty'
}

export type PasteIntent =
  | { kind: 'insert-inline-text'; text: string }
  | { kind: 'insert-segment-text'; text: string }
  | { kind: 'add-clipboard-image' }
  | { kind: 'add-path-image'; rawPath: string }
  | { kind: 'noop'; reason: 'empty' | 'blocked' | 'unsupported' }

export type PasteFeedbackReason = 'empty' | 'blocked' | 'unsupported'

export type PasteApplyResult =
  | { kind: 'inserted-inline-text' }
  | { kind: 'inserted-segment-text' }
  | { kind: 'added-clipboard-image' }
  | { kind: 'added-path-image' }
  | { kind: 'noop'; reason: PasteFeedbackReason }

export interface PasteEffectsResult {
  shouldReportInserted: boolean
  shouldBumpBulkInsertEpoch: boolean
  focusNudgeRequested: boolean
  feedbackReason?: PasteFeedbackReason
}
