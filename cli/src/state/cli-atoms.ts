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
  | { source: "env-local"; key: string; envVarName: string }
  | { source: "none" }

export const authSourceAtom = Atom.make<AuthSource>({ source: "none" })

/**
 * Recent chats overlay visibility (CLI only — web uses sidebar).
 */
export const showRecentChatsOverlayAtom = Atom.make(false)

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
 * Resolve auth source from environment variables.
 * Called once in the entry point, result passed as prop to CliApp.
 */
export function resolveEnvAuth(): AuthSource {
  const useLocal = process.env.MAGNITUDE_USE_LOCAL === "1" || process.env.MAGNITUDE_USE_LOCAL === "true"

  if (useLocal) {
    const localKey = process.env.MAGNITUDE_LOCAL_API_KEY
    if (localKey && localKey.trim()) {
      return { source: "env-local", key: localKey, envVarName: "MAGNITUDE_LOCAL_API_KEY" }
    }
  }

  const envKey = process.env.MAGNITUDE_API_KEY
  if (envKey && envKey.trim()) {
    return { source: "env", key: envKey, envVarName: "MAGNITUDE_API_KEY" }
  }

  return { source: "none" }
}

/**
 * Section anchor for the file viewer (markdown heading scroll target).
 * CLI-only display detail; the file path itself is the shared
 * selectedFilePathAtom from client-common.
 */
export const selectedFileSectionAtom = Atom.make<string | undefined>(undefined)
