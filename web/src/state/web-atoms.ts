/**
 * Web-only UI atoms — sidebar state that is specific to the web/desktop layout.
 *
 * Shared atoms (settings, usage, composer state, etc.) are imported from
 * `@magnitudedev/client-common`. This file holds only the atoms that have no
 * CLI counterpart.
 */
import { Atom } from "@effect-atom/atom-react"
import {
  selectedCwdAtom,
  settingsOpenAtom,
  usageOpenAtom,
  selectedFilePathAtom,
  composerDraftAtom,
  apiKeyVerifiedAtom,
  messageHistoryAtom,
  restoredQueuedInputTextAtom,
  bashModeAtom,
  nextEscWillKillAllAtom,
  composerHasContentAtom,
  pendingUserSubmitAtom,
  composerCanFocusAtom,
  bulkInsertEpochAtom,
  sessionActivationPromiseAtom,
  bashOutputsAtom,
} from "@magnitudedev/client-common"

// Re-export all shared atoms so existing web imports keep working
export {
  selectedCwdAtom,
  settingsOpenAtom,
  usageOpenAtom,
  selectedFilePathAtom,
  composerDraftAtom,
  apiKeyVerifiedAtom,
  messageHistoryAtom,
  restoredQueuedInputTextAtom,
  bashModeAtom,
  nextEscWillKillAllAtom,
  composerHasContentAtom,
  pendingUserSubmitAtom,
  composerCanFocusAtom,
  bulkInsertEpochAtom,
  sessionActivationPromiseAtom,
  bashOutputsAtom,
}

// ── Web-only atoms ──────────────────────────────────────────────

/**
 * Sidebar width in pixels.
 */
export const sidebarWidthAtom = Atom.make(260)

/**
 * Sidebar visibility for responsive overlay mode (≤640px).
 * null = not in responsive mode (sidebar always visible).
 * true/false = overlay sidebar visible/hidden.
 */
export const sidebarVisibleAtom = Atom.make<boolean | null>(null)

/**
 * Sidebar search query.
 */
export const sidebarSearchAtom = Atom.make("")

/**
 * Optional CWD filter for the sessions sidebar.
 * null = show sessions from every agent-host CWD.
 */
export const sidebarCwdFilterAtom = Atom.make<string | null>(null)
