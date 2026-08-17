import { Either, Schema } from "effect"
import { useSyncExternalStore } from "react"
import { injectPaletteCssVars } from "../styles/palette-css-vars"

export const AppearancePreferenceSchema = Schema.Literal(
  "system",
  "light",
  "dark"
)
export type AppearancePreference = typeof AppearancePreferenceSchema.Type
export type ResolvedAppearance = Exclude<AppearancePreference, "system">

const STORAGE_KEY = "magnitude.appearance"
const listeners = new Set<() => void>()
let preference: AppearancePreference = "system"
let initialized = false
let mediaQuery: MediaQueryList | null = null

const readStoredPreference = (): AppearancePreference => {
  try {
    const decoded = Schema.decodeUnknownEither(AppearancePreferenceSchema)(
      localStorage.getItem(STORAGE_KEY)
    )
    return Either.isRight(decoded) ? decoded.right : "system"
  } catch {
    return "system"
  }
}

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

export const initializeAppearance = (): void => {
  if (initialized) return
  initialized = true
  injectPaletteCssVars()
  preference = readStoredPreference()
  mediaQuery = matchMedia("(prefers-color-scheme: dark)")
  mediaQuery.addEventListener("change", () => {
    if (preference === "system") publish()
  })
  applyAppearance()
}

export const setAppearancePreference = (next: AppearancePreference): void => {
  if (preference === next) return
  preference = next
  try {
    if (next === "system") localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, next)
  } catch {
    // Storage can be unavailable in privacy-restricted browser contexts. The
    // live preference remains valid for the current renderer session.
  }
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
