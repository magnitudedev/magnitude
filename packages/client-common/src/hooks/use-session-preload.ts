import { Option, Effect } from "effect"
import { useMemo } from "react"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Sessions } from "@magnitudedev/sdk"
import {
  selectedCwdAtom,
  sessionCreateOptionsAtom,
} from "../state/session-atoms"
import { useAgentClient } from "../state/agent-client-context"
import { getDraftSessionOwnerId } from "./draft-session-owner"
import { useSelectedSessionId } from "../display-view-controller/hooks"


export function useSessionPreload(enabled = true): void {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const sessionCreateOptions = useAtomValue(sessionCreateOptionsAtom)
  const runtimeResult = useAtomValue(client.runtime)
  const runtimeReady = Result.isSuccess(runtimeResult)
  const preloadMutationAtom = useMemo(() => client.mutation(Sessions.PreloadSession), [client])
  const releaseMutationAtom = useMemo(() => client.mutation(Sessions.ReleaseSessionPreload), [client])
  const preloadSession = useAtomSet(preloadMutationAtom, { mode: "promise" })
  const releaseSessionPreload = useAtomSet(releaseMutationAtom, { mode: "promise" })

  const preloadAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!enabled || selectedSessionId || !selectedCwd || !runtimeReady) return
          const payload = {
            cwd: selectedCwd,
            options: sessionCreateOptions,
            draftOwnerId: Option.some(getDraftSessionOwnerId()),
          }
          const preloaded = yield* Effect.promise(() =>
            preloadSession(payload).catch((error: unknown) => {
              console.debug("[SessionPreload] preload failed:", error)
              return null
            }),
          )
          if (preloaded === null) return
          yield* Effect.addFinalizer(() =>
            Effect.promise(() =>
              releaseSessionPreload({ ...payload, sessionId: preloaded.sessionId }).catch(() => {
                // Best-effort cleanup; ACN also has owner replacement, TTL, and startup sweeps.
              }),
            ),
          )
        }),
      ),
    [enabled, selectedSessionId, selectedCwd, sessionCreateOptions, runtimeReady, preloadSession, releaseSessionPreload],
  )
  useAtomMount(preloadAtom)
}
