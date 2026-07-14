/**
 * Shared UI atoms — spec §6.3
 *
 * Client-local state shared across web, desktop, and CLI apps.
 * Uses Atom.make(initialValue) for plain writable local state.
 *
 * Web-only atoms (sidebar width/visibility/search) live in
 * `web/src/state/web-atoms.ts`. CLI-only atoms live in
 * `cli/src/state/cli-atoms.ts`.
 */
import { Atom } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { SessionOptions } from "@magnitudedev/sdk"
import type { BashResult } from "../utils/bash-executor"
import type { MentionAttachment } from "@magnitudedev/sdk"

/**
 * The agent-host CWD that will be used when creating a new session.
 * null = no working directory selected yet.
 */
export const selectedCwdAtom = Atom.make<string | null>(null)

/**
 * Settings panel open flag.
 */
export const settingsOpenAtom = Atom.make(false)

/**
 * Usage panel open flag.
 */
export const usageOpenAtom = Atom.make(false)

/**
 * File viewer panel: selected file path (null = closed).
 */
export const selectedFilePathAtom = Atom.make<string | null>(null)

/**
 * API key verified flag.
 * false = not verified (show login screen). true = key is set, show main app.
 */
export const apiKeyVerifiedAtom = Atom.make(false)

/**
 * Message history for composer up/down navigation.
 * Array of previously sent message texts, most recent first.
 */
export const messageHistoryAtom = Atom.make<string[]>([])

/**
 * Composer text content.
 * The composer reads and writes this atom directly. Restored queued input
 * writes here instead of triggering a reactive sync.
 */
export const composerTextAtom = Atom.make("")

/**
 * Composer attachment pills.
 * Restored queued input clears attachments by resetting this atom.
 */
export const composerAttachmentsAtom = Atom.make<MentionAttachment[]>([])

/**
 * Composer history navigation index.
 * -1 means not navigating history; restored input resets this to -1.
 */
export const composerHistoryIndexAtom = Atom.make(-1)

/**
 * Bash mode active flag for the composer.
 * When true, the composer sends commands via RunBash instead of SendMessage.
 */
export const bashModeAtom = Atom.make(false)

/**
 * "Next Esc will interrupt all workers" hint flag.
 * Set when the first Esc press closes the fork stack or no fork is open,
 * and the second Esc (within 400ms) will dispatch interrupt-all.
 * Cleared after a short timeout or when the hint is consumed.
 */
export const nextEscWillKillAllAtom = Atom.make(false)

// ── New shared atoms (Phase 0) ──────────────────────────────────

/**
 * True when the composer has non-empty content.
 * Used by the CLI to cancel autopilot countdown when the user starts typing.
 */
export const composerHasContentAtom = Atom.make(false)

/**
 * True while a user submit is pending (lazy session activation in progress).
 * Guards against concurrent session creation from rapid submits.
 */
export const pendingUserSubmitAtom = Atom.make(false)

/**
 * Holds the in-flight session activation promise during lazy session
 * creation. Enables concurrent submit deduplication — callers await the
 * same promise instead of creating duplicate sessions.
 */
export const sessionActivationPromiseAtom = Atom.make<Promise<string> | null>(null)

/**
 * Bash command outputs — shared so both apps can render bash output.
 * Each entry is a RunBashResult from the daemon.
 */
export const bashOutputsAtom = Atom.make<BashResult[]>([])

/**
 * Options applied when a client lazily creates a session (safeguard flags,
 * ATIF path, solo mode, system prompt override). The terminal app sets this
 * from CLI flags at startup; other clients leave the default (none).
 * Read by useComposerState's CreateSession path — atom-driven so the shared
 * hook has no optional parameters.
 */
export const sessionCreateOptionsAtom = Atom.make<Option.Option<SessionOptions>>(Option.none())
