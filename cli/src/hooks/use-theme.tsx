/**
 * Theme Hooks
 *
 * Simple hooks for accessing theme from zustand store
 */

import { create } from 'zustand'

import {
  activeThemeSettings,
  chatThemes,
  layerTheme,
  duplicateChatTheme,
} from '../utils/theme'

import type { ChatTheme, ThemeName } from '../types/theme-system'
import type { StoreApi, UseBoundStore } from 'zustand'

type ThemeState = {
  theme: ChatTheme
  setThemeName: (name: ThemeName) => void
  setTerminalDetectedBg: (color: string) => void
}

export let useThemeStateStore: UseBoundStore<StoreApi<ThemeState>> = (() => {
  throw new Error('useThemeStateStore not initialized')
}) as any
let hasThemeStoreInitialized = false

export function initThemeStore() {
  if (hasThemeStoreInitialized) {
    return
  }
  hasThemeStoreInitialized = true

  const initialTheme = layerTheme(
    duplicateChatTheme(chatThemes.dark),
    activeThemeSettings.customColors,
    activeThemeSettings.plugins,
    'dark',
  )

  useThemeStateStore = create<ThemeState>((set) => ({
    theme: initialTheme,
    setThemeName: (name: ThemeName) => {
      const baseTheme = name === 'light' ? chatThemes.light : chatThemes.dark
      const theme = layerTheme(
        duplicateChatTheme(baseTheme),
        activeThemeSettings.customColors,
        activeThemeSettings.plugins,
        name,
      )
      const currentDetectedBg = useThemeStateStore.getState().theme.terminalDetectedBg
      if (currentDetectedBg) {
        theme.terminalDetectedBg = currentDetectedBg
      }
      set({ theme })
    },
    setTerminalDetectedBg: (color: string) => {
      set((state) => ({ theme: { ...state.theme, terminalDetectedBg: color } }))
    },
  }))
}

export const useTheme = (): ChatTheme => {
  return useThemeStateStore((state) => state.theme)
}
