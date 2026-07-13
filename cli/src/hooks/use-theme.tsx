/**
 * Theme system — atom-based (replaces zustand store).
 *
 * The terminal color values stay CLI-specific (they map to OpenTUI TextAttributes
 * fg/bg codes). The store mechanism is @effect-atom/atom-react.
 */
import { Atom, useAtomValue, useAtomSet } from "@effect-atom/atom-react"

import {
  activeThemeSettings,
  chatThemes,
  layerTheme,
  duplicateChatTheme,
} from "../utils/theme"

import type { ChatTheme, ThemeName } from "../types/theme-system"

const initialTheme = layerTheme(
  duplicateChatTheme(chatThemes.dark),
  activeThemeSettings.customColors,
  activeThemeSettings.plugins,
  "dark",
)

export const themeAtom = Atom.make<ChatTheme>(initialTheme)

export function useTheme(): ChatTheme {
  return useAtomValue(themeAtom)
}

/**
 * Hook that returns theme setter functions.
 * Uses useAtomSet which is the effect-atom API for writing to atoms
 * (Atom.set returns an Effect requiring AtomRegistry — can't be called directly).
 */
export function useThemeActions() {
  const setTheme = useAtomSet(themeAtom)
  return {
    setThemeName(name: ThemeName): void {
      const baseTheme = name === "light" ? chatThemes.light : chatThemes.dark
      const theme = layerTheme(
        duplicateChatTheme(baseTheme),
        activeThemeSettings.customColors,
        activeThemeSettings.plugins,
        name,
      )
      setTheme(theme)
    },
    setTerminalDetectedBg(color: string): void {
      setTheme((prev) => ({ ...prev, terminalDetectedBg: color }))
    },
  }
}
