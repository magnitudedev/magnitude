import { Option, Effect } from "effect"
import { useMemo } from "react"
import { Atom, Result, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import {
  selectedCwdAtom,
  sessionCreateOptionsAtom,
} from "../state/session-atoms"
import { useAgentClient } from "../state/agent-client-context"
import { getDraftSessionOwnerId } from "./draft-session-owner"
import { useSelectedSessionId } from "../display-view-controller/hooks"

export function useSessionPreload(): void {
  const client = useAgentClient()
  const selectedSessionId = useSelectedSessionId()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const sessionCreateOptions = useAtomValue(sessionCreateOptionsAtom)
  const runtimeResult = useAtomValue(client.runtime)
  const runtimeReady = Result.isSuccess(runtimeResult)
  const preloadSession = useAtomSet(
    client.mutation("PreloadSession"),
    { mode: "promise" },
  )
  const releaseSessionPreload = useAtomSet(
    client.mutation("ReleaseSessionPreload"),
    { mode: "promise" },
  )

  const preloadAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (selectedSessionId || !selectedCwd || !runtimeReady) return
          const payload = {
            cwd: selectedCwd,
            options: sessionCreateOptions,
            draftOwnerId: Option.some(getDraftSessionOwnerId()),
          }
          yield* Effect.promise(() =>
            preloadSession({ payload, reactivityKeys: [] }).catch((error: unknown) => {
              console.debug("[SessionPreload] preload failed:", error)
            }),
          )
          yield* Effect.addFinalizer(() =>
            Effect.promise(() =>
              releaseSessionPreload({ payload, reactivityKeys: [] }).catch(() => {
                // Best-effort cleanup; ACN also has owner replacement, TTL, and startup sweeps.
              }),
            ),
          )
        }),
      ),
    [selectedSessionId, selectedCwd, sessionCreateOptions, runtimeReady, preloadSession, releaseSessionPreload],
  )
  useAtomMount(preloadAtom)
}
