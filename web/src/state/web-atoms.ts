/**
 * Web-only UI atoms — sidebar state that is specific to the web/desktop layout.
 *
 * Shared atoms (settings, composer state, etc.) are imported from
 * `@magnitudedev/client-common`. This file holds only the atoms that have no
 * CLI counterpart.
 */
import { Atom } from "@effect-atom/atom-react"
import type { ProjectId } from "@magnitudedev/sdk"
import type { ProjectFileRevision, ProjectRelativePath } from "@magnitudedev/sdk"
import {
  PROJECT_FILES_BROWSER_WIDTH,
  PROJECT_FILES_DOCUMENT_WIDTH,
} from "../lib/project-files-layout"
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

export const projectFilesPanelOpenAtom = Atom.keepAlive(Atom.make(false))
export interface ProjectFilesPanelWidths {
  readonly browser: number
  readonly document: number
}
export const projectFilesPanelWidthsAtom = Atom.keepAlive(
  Atom.make<ProjectFilesPanelWidths>({
    browser: PROJECT_FILES_BROWSER_WIDTH,
    document: PROJECT_FILES_DOCUMENT_WIDTH,
  }),
)
export const projectFileDirtyAtom = Atom.keepAlive(Atom.make(false))
export const projectFileDiscardIntentAtom = Atom.keepAlive(
  Atom.make<"back" | "close" | null>(null),
)
export interface ProjectFileDraft {
  readonly content: string
  readonly baseRevision: ProjectFileRevision
}
export const projectFileDraftsAtom = Atom.keepAlive(
  Atom.make<Readonly<Record<string, ProjectFileDraft>>>({}),
)
export interface SelectedProjectFile {
  readonly projectId: ProjectId
  readonly path: ProjectRelativePath
}
export const selectedProjectFileAtom = Atom.keepAlive(
  Atom.make<SelectedProjectFile | null>(null),
)
export const expandedProjectDirectoriesAtom = Atom.keepAlive(
  Atom.make<Readonly<Record<string, ReadonlySet<ProjectRelativePath>>>>({}),
)
