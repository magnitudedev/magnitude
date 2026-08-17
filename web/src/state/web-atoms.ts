/**
 * Web-only UI atoms — sidebar state that is specific to the web/desktop layout.
 *
 * Shared atoms (settings, composer state, etc.) are imported from
 * `@magnitudedev/client-common`. This file holds only the atoms that have no
 * CLI counterpart.
 */
import { Atom } from "@effect-atom/atom-react"
import type { ProjectId } from "@magnitudedev/sdk"
import {
  selectedCwdAtom,
  selectedFilePathAtom,
  composerTextAtom,
  composerAttachmentsAtom,
  composerHistoryIndexAtom,
  messageHistoryAtom,
  bashModeAtom,
  nextEscWillKillAllAtom,
  composerHasContentAtom,
  pendingUserSubmitAtom,
} from "@magnitudedev/client-common"

// Re-export all shared atoms so existing web imports keep working
export {
  selectedCwdAtom,
  selectedFilePathAtom,
  composerTextAtom,
  composerAttachmentsAtom,
  composerHistoryIndexAtom,
  messageHistoryAtom,
  bashModeAtom,
  nextEscWillKillAllAtom,
  composerHasContentAtom,
  pendingUserSubmitAtom,
}

// ── Web-only atoms ──────────────────────────────────────────────

/**
 * Sidebar width in pixels.
 */
export const sidebarWidthAtom = Atom.keepAlive(Atom.make(260))
export const sidebarCollapsedAtom = Atom.keepAlive(Atom.make(false))
export const collapsedProjectIdsAtom = Atom.keepAlive(
  Atom.make<ReadonlySet<ProjectId>>(new Set<ProjectId>()),
)
export type SettingsTab = "models" | "catalog" | "hardware"
export const settingsTabAtom = Atom.keepAlive(
  Atom.make<SettingsTab | null>(null)
)

/**
 * Whether the responsive sidebar overlay is open.
 * Wide layouts ignore this state because their sidebar is always docked.
 */
export const sidebarVisibleAtom = Atom.make(false)

/**
 * Sidebar search query.
 */
export const sidebarSearchAtom = Atom.keepAlive(Atom.make(""))
