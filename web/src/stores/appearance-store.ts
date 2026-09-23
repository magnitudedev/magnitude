import { AppearancePreference as AppearancePreferenceSchema } from "@magnitudedev/sdk/desktop-host"
export { AppearancePreferenceSchema }
import { useSyncExternalStore } from "react"
import { injectPaletteCssVars } from "../styles/palette-css-vars"

export type AppearancePreference = typeof AppearancePreferenceSchema.Type
export type ResolvedAppearance = Exclude<AppearancePreference, "system">

const listeners = new Set<() => void>()
let preference: AppearancePreference = "system"
let initialized = false
let mediaQuery: MediaQueryList | null = null

export const getResolvedAppearance = (): ResolvedAppearance => {
  if (preference !== "system") return preference
  const prefersDark =
    mediaQuery?.matches ??
    (typeof matchMedia === "function" &&
      matchMedia("(prefers-color-scheme: dark)").matches)
  return prefersDark ? "dark" : "light"
}

const applyAppearance = (): void => {
  const resolved = getResolvedAppearance()
  document.documentElement.dataset.theme = resolved
  document.documentElement.style.colorScheme = resolved
}

const publish = (): void => {
  applyAppearance()
  listeners.forEach((listener) => listener())
}

export const initializeAppearance = (initial: AppearancePreference = "system"): void => {
  if (initialized) return
  initialized = true
  injectPaletteCssVars()
  preference = initial
  mediaQuery = matchMedia("(prefers-color-scheme: dark)")
  mediaQuery.addEventListener("change", () => {
    if (preference === "system") publish()
  })
  applyAppearance()
}

export const setAppearancePreference = (next: AppearancePreference): void => {
  if (preference === next) return
  preference = next
  publish()
}

export const getAppearancePreference = (): AppearancePreference => preference

export const subscribeAppearance = (listener: () => void): (() => void) => {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export const useAppearancePreference = (): AppearancePreference =>
  useSyncExternalStore(
    subscribeAppearance,
    getAppearancePreference,
    () => "system"
  )

export const useResolvedAppearance = (): ResolvedAppearance =>
  useSyncExternalStore(
    subscribeAppearance,
    getResolvedAppearance,
    () => "light",
  )
