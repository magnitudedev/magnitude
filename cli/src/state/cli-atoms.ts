/**
 * CLI-only atoms — state specific to the terminal app.
 *
 * Shared atoms live in `@magnitudedev/client-common`.
 * Web-only atoms live in `web/src/state/web-atoms.ts`.
 */
import { Atom } from "@effect-atom/atom-react"

/**
 * Auth source — how the API key was resolved.
 * CLI-only: web has no env vars.
 */
export type AuthSource =
  | { source: "config" }
  | { source: "env"; key: string; envVarName: string }
  | { source: "none" }

export const authSourceAtom = Atom.make<AuthSource>({ source: "none" })

/**
 * Recent chats overlay visibility (CLI only — web uses sidebar).
 */
export const showRecentChatsOverlayAtom = Atom.make(false)

/** CLI local-model setup route. */
export type ModelSetupRoute = "closed" | "local"
export const modelSetupRouteAtom = Atom.make<ModelSetupRoute>("closed")

/** Standalone Magnitude Cloud setup overlay, opened by /cloud. */
export const cloudModelsOpenAtom = Atom.make(false)

/**
 * Autopilot atoms — disabled but kept for potential future re-enablement.
 * Not wired to any active logic. Components exist but are not rendered.
 */
export const autopilotEnabledAtom = Atom.make(false)
export const autopilotGeneratingAtom = Atom.make(false)

export interface AutopilotCountdown {
  seconds: number
  preview: string | null
}
export const autopilotCountdownAtom = Atom.make<AutopilotCountdown>({
  seconds: 0,
  preview: null,
})

export const autopilotRetainedContentAtom = Atom.make<string | null>(null)

/**
 * Section anchor for the file viewer (markdown heading scroll target).
 * CLI-only display detail; the file path itself is the shared
 * selectedFilePathAtom from client-common.
 */
export const selectedFileSectionAtom = Atom.make<string | undefined>(undefined)
