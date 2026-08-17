import { Option, Effect } from "effect"
import { useMemo } from "react"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  selectedCwdAtom,
  selectedProjectIdAtom,
  sessionCreateOptionsAtom,
} from "../state/session-atoms"
import { useAgentClient } from "../state/agent-client-context"
import { getDraftSessionOwnerId } from "./draft-session-owner"
import { useSelectedSessionId } from "../display-view-controller/hooks"

export function useSessionPreload(enabled = true): void {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const selectedProjectId = useAtomValue(selectedProjectIdAtom)
  const sessionCreateOptions = useAtomValue(sessionCreateOptionsAtom)
  const runtimeResult = useAtomValue(client.rpc.runtime)
  const runtimeReady = Result.isSuccess(runtimeResult)
  const preloadMutationAtom = useMemo(() => client.rpc.mutation("PreloadSession"), [client])
  const releaseMutationAtom = useMemo(() => client.rpc.mutation("ReleaseSessionPreload"), [client])
  const preloadSession = useAtomSet(preloadMutationAtom, { mode: "promise" })
  const releaseSessionPreload = useAtomSet(releaseMutationAtom, { mode: "promise" })

  const preloadAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!enabled || selectedSessionId || !selectedCwd || !runtimeReady) return
          const payload = {
            cwd: selectedCwd,
            projectId: Option.fromNullable(selectedProjectId),
            options: sessionCreateOptions,
            draftOwnerId: Option.some(getDraftSessionOwnerId()),
          }
          const preloaded = yield* Effect.promise(() =>
            preloadSession({ payload, reactivityKeys: [] }).catch((error: unknown) => {
              console.debug("[SessionPreload] preload failed:", error)
              return null
            }),
          )
          if (preloaded === null) return
          yield* Effect.addFinalizer(() =>
            Effect.promise(() =>
              releaseSessionPreload({
                payload: { ...payload, sessionId: preloaded.sessionId },
                reactivityKeys: [],
              }).catch(() => {
                // Best-effort cleanup; ACN also has owner replacement, TTL, and startup sweeps.
              }),
            ),
          )
        }),
      ),
    [enabled, selectedSessionId, selectedCwd, selectedProjectId, sessionCreateOptions, runtimeReady, preloadSession, releaseSessionPreload],
  )
  useAtomMount(preloadAtom)
}
