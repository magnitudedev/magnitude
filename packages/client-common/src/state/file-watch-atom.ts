/**
 * File watch reactivity bridge — subscribes to WatchFile for the selected
 * file and invalidates the "files" reactivity key on each event.
 *
 * This makes `reactivityKeys: ["files"]` on ReadFile queries actually work:
 * when a file changes on disk, the reactivity key is invalidated, and any
 * atom query with that key re-executes.
 *
 * Uses `useAtomMount` — the atom runtime provides the `Reactivity` service.
 * When the selected file changes, `useMemo` creates a new atom, `useAtomMount`
 * mounts the new one and unmounts the old (interrupting the old watch stream).
 * No manual fiber management.
 */
import { useMemo } from "react"
import { useAtomMount, useAtomValue } from "@effect-atom/atom-react"
import { Effect, Stream, Cause } from "effect"
import * as Reactivity from "@effect/experimental/Reactivity"
import {
  type WatchFileWireEvent,
  type WatchFileEvent,
} from "@magnitudedev/sdk"
import { useAgentClient } from "./agent-client-context"
import { selectedCwdAtom, selectedFilePathAtom } from "./session-atoms"

const isFileEvent = (event: WatchFileWireEvent): event is WatchFileEvent =>
  !("_tag" in event)

/**
 * Subscribe to WatchFile for the selected file and invalidate the "files"
 * reactivity key on each event. Mount this once at the app root.
 */
export function useFileWatchBridge(): void {
  const client = useAgentClient()
  const selectedCwd = useAtomValue(selectedCwdAtom)
  const filePath = useAtomValue(selectedFilePathAtom)

  const watchAtom = useMemo(
    () =>
      client.runtime.atom(
        Effect.gen(function* () {
          if (!selectedCwd || !filePath) return

          const rpc = yield* client

          yield* rpc("WatchFile", { cwd: selectedCwd, path: filePath }).pipe(
            Stream.filter(isFileEvent),
            Stream.tap(() => Reactivity.invalidate(["files"])),
            Stream.runDrain,
          )
        }).pipe(
          Effect.catchAllCause((cause) =>
            Cause.isInterruptedOnly(cause)
              ? Effect.void
              : Effect.logError(`[FileWatch] ${Cause.pretty(cause)}`),
          ),
        ),
      ),
    [client, selectedCwd, filePath],
  )

  useAtomMount(watchAtom)
}
