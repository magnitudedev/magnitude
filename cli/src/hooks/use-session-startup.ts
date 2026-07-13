/**
 * Session startup — resolves the initial session and handles one-shot
 * session-level work, all reactively off atoms. Owns:
 *
 * 1. Initial session resolution: --resume <id>, --resume latest (query
 *    ListSessions), or new (null). Selects the display controller session once.
 * 2. Session attach side effects on every session change: skill command
 *    registration, session logger, terminal title.
 * 3. One-shot --prompt / --goal send after the first display arrives.
 *
 * No useEffect: one-shot work is guarded by refs (spec §5.6 rule 11),
 * reactions are render-time ref-compares against atom values — the same
 * pattern the web app uses for responsive sidebar sync.
 */
import { useRef, useCallback } from 'react'
import { Effect, Option, Runtime } from 'effect'
import { useAtomValue, useAtomSet, Result } from '@effect-atom/atom-react'
import { useRenderer } from '@opentui/react'
import { logger, initLogger } from '@magnitudedev/logger'
import {
  useAgentClient,
  useDisplayState,
  selectedCwdAtom,
  sessionCreateOptionsAtom,
  useDisplayViewControllerCore,
  useHasReceivedDisplay,
  useSelectedSessionId,
  pendingUserSubmitAtom,
  loadSkillCommands,
  registerSkillCommands,
  getDraftSessionOwnerId,
} from '@magnitudedev/client-common'
import { setLastSessionId } from '../state/last-session'

export type SessionStart =
  | { _tag: 'new' }
  | { _tag: 'latest' }
  | { _tag: 'resume'; sessionId: string }

export interface SessionStartupParams {
  sessionStart: SessionStart
  initialPrompt: string | undefined
  goal: string | undefined
}

export function useSessionStartup({ sessionStart, initialPrompt, goal }: SessionStartupParams): void {
  const client = useAgentClient()
  const renderer = useRenderer()
  const controller = useDisplayViewControllerCore()
  const sessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const sessionCreateOptions = useAtomValue(sessionCreateOptionsAtom)
  const hasReceivedDisplay = useHasReceivedDisplay()
  const setPendingUserSubmit = useAtomSet(pendingUserSubmitAtom)
  const runtimeResult = useAtomValue(client.runtime)

  const runRpc = useCallback(<A, E, R>(effect: Effect.Effect<A, E, R>): Promise<A> => {
    if (!Result.isSuccess(runtimeResult)) {
      return Promise.reject(new Error('AgentClient runtime not ready'))
    }
    return Runtime.runPromise(runtimeResult.value)(effect as Effect.Effect<A, E, never>)
  }, [runtimeResult])

  // ── 1. Initial session resolution — one-shot, waits for runtime ready ──
  const resolvedStartRef = useRef(false)
  if (!resolvedStartRef.current && Result.isSuccess(runtimeResult)) {
    resolvedStartRef.current = true
    if (sessionStart._tag === 'resume') {
      controller.selectSession(sessionStart.sessionId)
    } else if (sessionStart._tag === 'latest') {
      void runRpc(Effect.flatMap(client, (c) =>
        c('ListSessions', { cwd: Option.some(process.cwd()), query: Option.none(), cursor: Option.none(), limit: 1 })
      )).then((result) => {
        const latest = result.items[0]
        if (latest) controller.selectSession(latest.sessionId)
        // No sessions to resume — the null-session empty state stands.
      }).catch((error) => {
        logger.error(
          { error: error instanceof Error ? error.message : String(error) },
          'Failed to resolve latest session'
        )
      })
    }
    // 'new' leaves the controller without a selected session; the first send
    // lazily creates and selects the session.
  }

  // ── 2. Session attach side effects — react to session change ──────────
  const prevSessionIdRef = useRef<string | null | undefined>(undefined)
  if (prevSessionIdRef.current !== sessionId) {
    prevSessionIdRef.current = sessionId
    setLastSessionId(sessionId)
    if (sessionId) {
      initLogger(sessionId)
    }
  }

  // ── 2b. Skills loading — react to cwd change (skills are cwd-scoped) ──
  const prevCwdRef = useRef<string | null | undefined>(undefined)
  if (prevCwdRef.current !== selectedCwd) {
    prevCwdRef.current = selectedCwd
    if (selectedCwd) {
      loadSkillCommands({
        listSkills: (cwd: string) => runRpc(Effect.flatMap(client, (c) => c('ListSkills', { cwd }))),
      }, selectedCwd).then((commands) => {
        if (commands.length > 0) registerSkillCommands(commands)
      }).catch((error) => {
        logger.error(
          { error: error instanceof Error ? error.message : String(error) },
          'Failed to load skills'
        )
      })
    }
  }

  // ── Terminal title tracks the display projection's session title ───────
  // The title is already streamed in display state — no RPC needed.
  const sessionTitle = useDisplayState((state) => state.session.title)
  const prevTitleRef = useRef<string | null | undefined>(undefined)
  const effectiveTitle = sessionId ? (sessionTitle ?? 'Magnitude') : 'Magnitude'
  if (prevTitleRef.current !== effectiveTitle) {
    prevTitleRef.current = effectiveTitle
    renderer.setTerminalTitle(effectiveTitle)
  }

  // ── 3. One-shot --prompt / --goal after first display ─────────────────
  // With an existing session, sends into it. With none, creates the session
  // with the prompt/goal as initial content (the lazy-activation path).
  const initialWorkSentRef = useRef(false)
  if (hasReceivedDisplay && !initialWorkSentRef.current && Result.isSuccess(runtimeResult)) {
    const goalObjective = goal?.trim()
    const prompt = initialPrompt?.trim()
    if (goalObjective || prompt) {
      initialWorkSentRef.current = true
      setPendingUserSubmit(true)
      const initial = goalObjective
        ? { _tag: 'goal' as const, objective: goalObjective }
        : {
            _tag: 'message' as const,
            messageId: Option.none<string>(),
            content: prompt!,
            visibleMessage: Option.some(prompt!),
            taskMode: false,
            attachments: [],
          }
      const work = sessionId
        ? (goalObjective
            ? runRpc(Effect.flatMap(client, (c) => c('StartGoal', { sessionId, objective: goalObjective })))
            : runRpc(Effect.flatMap(client, (c) =>
                c('SendMessage', {
                  sessionId,
                  messageId: Option.none<string>(),
                  content: prompt!,
                  visibleMessage: Option.some(prompt!),
                  taskMode: false,
                  attachments: [],
                })
              )))
        : runRpc(Effect.flatMap(client, (c) =>
            c('CreateSession', {
              cwd: selectedCwd ?? process.cwd(),
              sessionId: Option.none(),
              initial: Option.some(initial),
              options: sessionCreateOptions,
              draftOwnerId: Option.some(getDraftSessionOwnerId()),
            })
          )).then((result) => {
            if (result._tag === 'created') {
              controller.selectSession(result.metadata.sessionId)
            } else if (result._tag === 'created_message_failed') {
              controller.selectSession(result.sessionId)
            }
            // failed: do nothing — the catch handler will log the error
            setPendingUserSubmit(false)
          })
      void work.catch((error) => {
        logger.error(
          { error: error instanceof Error ? error.message : String(error) },
          'Failed to send initial prompt/goal'
        )
        setPendingUserSubmit(false)
      })
    }
  }
}
