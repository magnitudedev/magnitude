import { Either, Schema } from "effect"
import { AppearancePreferenceSchema, initializeAppearance, setAppearancePreference, type AppearancePreference } from "./appearance-store"

const key = "magnitude.appearance"

export const initializeBrowserAppearance = (): void => {
  let preference: AppearancePreference = "system"
  try {
    const decoded = Schema.decodeUnknownEither(AppearancePreferenceSchema)(localStorage.getItem(key))
    if (Either.isRight(decoded)) preference = decoded.right
  } catch { /* Browser storage may be unavailable. */ }
  initializeAppearance(preference)
}

export const setBrowserAppearancePreference = (preference: AppearancePreference): void => {
  try {
    if (preference === "system") localStorage.removeItem(key)
    else localStorage.setItem(key, preference)
  } catch { /* Keep the current page usable when browser storage is unavailable. */ }
  setAppearancePreference(preference)
}
