/**
 * Menu actions core — shared platform menu subscription.
 *
 * Subscribes to platform.onMenuAction (Electron menu). Browser-mode
 * keyboard shortcuts and Esc handling are app-specific — each app has
 * its own keyboard handler. This hook only handles the platform menu
 * subscription.
 *
 * Uses useAtomMount — effect-atom's lifecycle API. No useEffect.
 */
import { useMemo } from "react"
import { Effect } from "effect"
import { Atom, useAtomSet, useAtomMount } from "@effect-atom/atom-react"
import { usePlatform } from "../platform/platform-context"
import { settingsOpenAtom } from "../state/session-atoms"
import { useSessionActions } from "./use-session-actions"
import { useDisplayViewController } from "../display-view-controller/hooks"

export function useMenuActionsCore(): void {
  const platform = usePlatform()
  const { startNewSession } = useSessionActions()
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const { displayMode, togglePresentationMode } = useDisplayViewController()

  const menuAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          if (!platform.onMenuAction) return
          const unsub = platform.onMenuAction((action) => {
            switch (action._tag) {
              case "new-session":
                startNewSession()
                break
              case "toggle-sidebar-search":
                // App-specific — dispatch a custom event each app listens for
                if (typeof window !== "undefined") {
                  window.dispatchEvent(new CustomEvent("__magnitude:focus-search"))
                }
                break
              case "toggle-transcript-mode":
                togglePresentationMode()
                break
              case "open-settings":
                setSettingsOpen(true)
                break
              case "quit":
                platform.quit?.()
                break
            }
          })
          yield* Effect.addFinalizer(() => Effect.sync(unsub))
        }),
      ),
    [platform, displayMode, startNewSession, togglePresentationMode, setSettingsOpen],
  )

  useAtomMount(menuAtom)
}
