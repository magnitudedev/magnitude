/**
 * Session startup — resolves the initial session and handles one-shot
 * session-level work reactively off atoms. Owns:
 *
 * 1. Initial session resolution: --resume <id>, --resume latest (query
 *    ListSessions), or new (null). Selects the display controller session once.
 * 2. Session attach side effects on every session change: skill command
 *    registration, session logger, terminal title.
 * 3. One-shot --prompt / --goal send after the first display arrives.
 *
 * Reads use client.query() (declarative). Mutations use useAtomSet with
 * { mode: "promise" } where the return value is needed. No runRpc, no
 * Runtime.runPromise, no as casts.
 */
import { useMemo } from 'react'
import { Effect, Option } from 'effect'
import { useAtomValue, useAtomSet, Result, Atom, useAtomMount } from '@effect-atom/atom-react'
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
  registerSkillCommands,
  getDraftSessionOwnerId,
} from '@magnitudedev/client-common'
import { setLastSessionId } from '../state/last-session'
import type { ListSessionsResult } from '@magnitudedev/sdk'
import type { SkillListEntry } from '@magnitudedev/sdk'

export type SessionStart =
  | { _tag: 'new' }
  | { _tag: 'latest' }
  | { _tag: 'resume'; sessionId: string }

export interface SessionStartupParams {
  sessionStart: SessionStart
  initialPrompt: string | undefined
  goal: string | undefined
  modelsConfigured: boolean
}

const idleAtom = Atom.make(() => null)

export function useSessionStartup({ sessionStart, initialPrompt, goal, modelsConfigured }: SessionStartupParams): void {
  const client = useAgentClient()
  const renderer = useRenderer()
  const controller = useDisplayViewControllerCore()
  const sessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const sessionCreateOptions = useAtomValue(sessionCreateOptionsAtom)
  const hasReceivedDisplay = useHasReceivedDisplay()
  const setPendingUserSubmit = useAtomSet(pendingUserSubmitAtom)
  const runtimeResult = useAtomValue(client.runtime)
  const sessionTitle = useDisplayState((state) => state.session.title)

  // ── 1. Latest session query — declarative, only for --latest ──────────
  const latestSessionAtom = useMemo(
    () =>
      sessionStart._tag === 'latest' && Result.isSuccess(runtimeResult)
        ? client.query('ListSessions', {
            cwd: Option.some(process.cwd()),
            query: Option.none(),
            cursor: Option.none(),
            limit: 1,
          }, { reactivityKeys: ['sessions'] })
        : idleAtom,
    [client, sessionStart, runtimeResult],
  )
  const latestSessionResult = useAtomValue(latestSessionAtom)

  // Select the latest session once the query resolves
  const initSessionAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (sessionStart._tag === 'resume') {
            controller.selectSession(sessionStart.sessionId)
          } else if (sessionStart._tag === 'latest' && latestSessionResult && Result.isSuccess(latestSessionResult)) {
            const latest = (latestSessionResult.value as ListSessionsResult).items[0]
            if (latest) controller.selectSession(latest.sessionId)
          }
        }),
      ),
    [sessionStart, controller, latestSessionResult],
  )
  useAtomMount(initSessionAtom)

  // ── 2. Session attach side effects — react to session change ──────────
  const sessionAttachAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          setLastSessionId(sessionId)
          if (sessionId) {
            initLogger(sessionId)
          }
        }),
      ),
    [sessionId],
  )
  useAtomMount(sessionAttachAtom)

  // ── 2b. Skills query — declarative, reacts to cwd change ──────────────
  const skillsAtom = useMemo(
    () =>
      selectedCwd && Result.isSuccess(runtimeResult)
        ? client.query('ListSkills', { cwd: selectedCwd }, { reactivityKeys: ['skills'] })
        : idleAtom,
    [client, selectedCwd, runtimeResult],
  )
  const skillsResult = useAtomValue(skillsAtom)

  // Register/unregister skill commands based on the query result
  const skillRegistrationAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!skillsResult || !Result.isSuccess(skillsResult)) return
          const entries: readonly SkillListEntry[] = (skillsResult.value as { skills?: SkillListEntry[] }).skills ?? []
          const commands = entries.map((s) => ({
            id: s.name,
            label: s.name,
            description: s.description,
            source: 'skill' as const,
            skillPath: s.path,
          }))
          if (commands.length > 0) registerSkillCommands(commands)
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => {
              registerSkillCommands([])
            }),
          )
        }),
      ),
    [skillsResult],
  )
  useAtomMount(skillRegistrationAtom)

  // ── Terminal title tracks the display projection's session title ───────
  const titleAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          const title = sessionId ? (sessionTitle ?? 'Magnitude') : 'Magnitude'
          renderer.setTerminalTitle(title)
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => {
              renderer.setTerminalTitle('Magnitude')
            }),
          )
        }),
      ),
    [renderer, sessionId, sessionTitle],
  )
  useAtomMount(titleAtom)

  // ── 3. One-shot --prompt / --goal — true mutations needing return value
  const startGoalMutation = useAtomSet(client.mutation('StartGoal'), { mode: 'promise' })
  const sendMessageMutation = useAtomSet(client.mutation('SendMessage'), { mode: 'promise' })
  const createSessionMutation = useAtomSet(client.mutation('CreateSession'), { mode: 'promise' })

  const initialWorkAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!hasReceivedDisplay || !Result.isSuccess(runtimeResult)) return
          if (!modelsConfigured) return
          const goalObjective = goal?.trim()
          const prompt = initialPrompt?.trim()
          if (!goalObjective && !prompt) return

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
            ? goalObjective
              ? startGoalMutation({
                  payload: { sessionId, objective: goalObjective },
                  reactivityKeys: ['sessions'],
                })
              : sendMessageMutation({
                  payload: {
                    sessionId,
                    messageId: Option.none<string>(),
                    content: prompt!,
                    visibleMessage: Option.some(prompt!),
                    taskMode: false,
                    attachments: [],
                  },
                  reactivityKeys: ['sessions'],
                })
            : createSessionMutation({
                payload: {
                  cwd: selectedCwd ?? process.cwd(),
                  sessionId: Option.none(),
                  initial: Option.some(initial),
                  options: sessionCreateOptions,
                  draftOwnerId: Option.some(getDraftSessionOwnerId()),
                },
                reactivityKeys: ['sessions'],
              }).then((result) => {
                if (result._tag === 'created') {
                  controller.selectSession(result.metadata.sessionId)
                } else if (result._tag === 'created_message_failed') {
                  controller.selectSession(result.sessionId)
                }
                setPendingUserSubmit(false)
              })

          yield* Effect.promise(() =>
            work.catch((error) => {
              logger.error(
                { error: error instanceof Error ? error.message : String(error) },
                'Failed to send initial prompt/goal',
              )
              setPendingUserSubmit(false)
            }),
          )
        }),
      ),
    [
      hasReceivedDisplay,
      modelsConfigured,
      goal,
      initialPrompt,
      sessionId,
      selectedCwd,
      runtimeResult,
      sessionCreateOptions,
      controller,
      setPendingUserSubmit,
      startGoalMutation,
      sendMessageMutation,
      createSessionMutation,
    ],
  )
  useAtomMount(initialWorkAtom)
}
