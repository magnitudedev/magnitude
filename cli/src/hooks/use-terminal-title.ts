import { useMemo } from 'react'
import { Atom, useAtomMount } from '@effect-atom/atom-react'
import { Effect } from 'effect'

interface TerminalTitleRenderer {
  readonly setTerminalTitle: (title: string) => void
}

export function terminalTitleForSession(
  sessionId: string | null,
  sessionTitle: string | null | undefined,
): string {
  return sessionId ? (sessionTitle ?? 'Magnitude') : 'Magnitude'
}

/**
 * Keep the terminal title synchronized with the selected session.
 *
 * Updating and cleanup intentionally use separate atom lifecycles. A title
 * change replaces the update atom, while the cleanup atom remains mounted;
 * otherwise the previous title atom's finalizer can reset a newer title.
 */
export function useTerminalTitle(
  renderer: TerminalTitleRenderer,
  sessionId: string | null,
  sessionTitle: string | null | undefined,
): void {
  const title = terminalTitleForSession(sessionId, sessionTitle)

  const updateTitleAtom = useMemo(
    () => Atom.make(Effect.sync(() => renderer.setTerminalTitle(title))),
    [renderer, title],
  )
  useAtomMount(updateTitleAtom)

  const cleanupTitleAtom = useMemo(
    () => Atom.make(Effect.addFinalizer(() =>
      Effect.sync(() => renderer.setTerminalTitle('Magnitude')),
    )),
    [renderer],
  )
  useAtomMount(cleanupTitleAtom)
}
