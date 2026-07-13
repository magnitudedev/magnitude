/**
 * Menu actions hook — spec §14.4
 *
 * Subscribes to platform menu actions (Electron) and local keyboard shortcuts.
 * In browser mode (platform.id === "web"), there's no Electron menu, so we
 * handle Cmd/Ctrl+N, Cmd/Ctrl+R, Cmd/Ctrl+T, Cmd/Ctrl+, directly via a
 * document-level keydown listener. Esc handling is always local.
 *
 * Uses useAtomMount — effect-atom's lifecycle API. No useEffect.
 */
import { useMemo } from "react"
import { Effect } from "effect"
import { Atom, useAtomSet, useAtomValue, useAtomMount } from "@effect-atom/atom-react"
import { useDisplayViewController, usePlatform, useSessionActions } from "@magnitudedev/client-common"
import {
  settingsOpenAtom,
  usageOpenAtom,
  nextEscWillKillAllAtom,
} from "@magnitudedev/client-common"
import { sidebarSearchAtom } from "../state/web-atoms"

/** Check if the platform uses Cmd (macOS) vs Ctrl (other). */
function isModKey(e: KeyboardEvent): boolean {
  return e.metaKey || e.ctrlKey
}

/**
 * Subscribe to platform menu actions and wire them to UI state.
 * Also handles local keyboard shortcuts (Esc, Esc×2, and browser-mode
 * Cmd/Ctrl+N, R, T, comma).
 */
export function useMenuActions(): void {
  const platform = usePlatform()
  const { startNewSession } = useSessionActions()
  const setSearchQuery = useAtomSet(sidebarSearchAtom)
  const setSettingsOpen = useAtomSet(settingsOpenAtom)
  const setUsageOpen = useAtomSet(usageOpenAtom)
  const setNextEscWillKillAll = useAtomSet(nextEscWillKillAllAtom)
  const { popFork, togglePresentationMode, displayMode, expandedForkStack } = useDisplayViewController()
  const settingsOpen = useAtomValue(settingsOpenAtom)
  const usageOpen = useAtomValue(usageOpenAtom)

  // Menu subscription atom — subscribes on mount, unsubscribes on dispose.
  // Only present in desktop mode (platform.onMenuAction is undefined in browser).
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
                setSearchQuery("")
                window.dispatchEvent(new CustomEvent("__magnitude:focus-search"))
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
    [platform, displayMode, startNewSession, setSearchQuery, togglePresentationMode, setSettingsOpen],
  )

  // Keyboard handler atom — Esc handling (always) + browser-mode shortcuts.
  // In desktop mode, Electron's application menu handles Cmd/Ctrl+N, R, T, comma
  // and dispatches them through onMenuAction. In browser mode, we handle them here.
  const keyboardAtom = useMemo(
    () =>
      Atom.make(
        Effect.gen(function* () {
          let lastEscTime = 0
          const handler = (e: KeyboardEvent) => {
            // ── Escape handling (all platforms) ──
            if (e.key === "Escape") {
              const now = Date.now()
              const isDoubleEsc = now - lastEscTime < 400
              lastEscTime = now

              // Close settings/usage panel first
              if (settingsOpen || usageOpen) {
                setSettingsOpen(false)
                setUsageOpen(false)
                e.preventDefault()
                return
              }

              if (expandedForkStack.length > 0) {
                popFork()
                e.preventDefault()
                return
              }

              if (isDoubleEsc) {
                setNextEscWillKillAll(false)
                window.dispatchEvent(new CustomEvent("__magnitude:interrupt-all"))
                e.preventDefault()
              } else {
                // First Esc — show hint that next Esc will interrupt all
                setNextEscWillKillAll(true)
                // Clear hint after 400ms if no second Esc comes
                setTimeout(() => {
                  setNextEscWillKillAll(false)
                }, 400)
              }
              return
            }

            // ── Browser-mode global shortcuts (spec §14.4) ──
            // Desktop mode uses Electron's application menu → onMenuAction.
            // Only intercept here in browser mode.
            if (platform.id === "web" && isModKey(e) && !e.shiftKey && !e.altKey) {
              switch (e.key.toLowerCase()) {
                case "n":
                  // Cmd/Ctrl+N → new session
                  e.preventDefault()
                  startNewSession()
                  break
                case "r":
                  // Cmd/Ctrl+R → focus sidebar search
                  e.preventDefault()
                  setSearchQuery("")
                  window.dispatchEvent(new CustomEvent("__magnitude:focus-search"))
                  break
                case "t":
                  // Cmd/Ctrl+T → toggle transcript mode
                  e.preventDefault()
                  togglePresentationMode()
                  break
                case ",":
                  // Cmd/Ctrl+, → open settings
                  e.preventDefault()
                  setSettingsOpen(true)
                  break
              }
            }
          }
          window.addEventListener("keydown", handler)
          yield* Effect.addFinalizer(() =>
            Effect.sync(() => window.removeEventListener("keydown", handler)),
          )
        }),
      ),
    [expandedForkStack.length, platform.id, displayMode, settingsOpen, usageOpen, popFork, startNewSession, setSearchQuery, togglePresentationMode, setSettingsOpen, setUsageOpen, setNextEscWillKillAll],
  )

  useAtomMount(menuAtom)
  useAtomMount(keyboardAtom)
}
