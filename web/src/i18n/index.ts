import { useCallback, useEffect, useState } from "react"
import { en } from "./catalogs/en"
import { es } from "./catalogs/es"
import { zhCN } from "./catalogs/zh-CN"
import { hi } from "./catalogs/hi"
import { ur } from "./catalogs/ur"
import { bn } from "./catalogs/bn"

export const LANGUAGES = ["en", "es", "zh-CN", "hi", "ur", "bn"] as const
export type Language = typeof LANGUAGES[number]
export type MessageKey = keyof typeof en
export const STORAGE_KEY = "magnitude.language"
const catalogs: Record<Language, Record<MessageKey, string>> = { en, es, "zh-CN": zhCN, hi, ur, bn }

export function isLanguage(value: unknown): value is Language {
  return typeof value === "string" && (LANGUAGES as readonly string[]).includes(value)
}

/** localStorage is shared by the desktop renderer and browser client, scoped to each app/browser profile. */
export function readLanguage(storage: Pick<Storage, "getItem"> | undefined = safeStorage()): Language {
  try {
    const stored = storage?.getItem(STORAGE_KEY)
    return isLanguage(stored) ? stored : "en"
  } catch {
    return "en"
  }
}

function safeStorage(): Storage | undefined {
  try {
    return typeof localStorage === "undefined" ? undefined : localStorage
  } catch {
    return undefined
  }
}

export function translate(language: Language, key: MessageKey): string {
  return catalogs[language][key] ?? en[key]
}

const englishMessageKeys = new Map<string, MessageKey>()
for (const [key, value] of Object.entries(en) as Array<[MessageKey, string]>) {
  if (!englishMessageKeys.has(value)) englishMessageKeys.set(value, key)
}

/** Translate exact known UI messages that arrive from service/host layers; preserve unknown diagnostics. */
export function translateKnownMessage(message: string, language: Language = readLanguage()): string {
  const key = englishMessageKeys.get(message)
  return key ? translate(language, key) : message
}

const changeEvent = "magnitude-language-change"

export function useI18n() {
  const [language, setLanguageState] = useState<Language>(() => readLanguage())
  useEffect(() => {
    document.documentElement.lang = language
    document.documentElement.dir = language === "ur" ? "rtl" : "ltr"
  }, [language])
  useEffect(() => {
    const sync = () => setLanguageState(readLanguage())
    window.addEventListener("storage", sync)
    window.addEventListener(changeEvent, sync)
    return () => {
      window.removeEventListener("storage", sync)
      window.removeEventListener(changeEvent, sync)
    }
  }, [])
  const setLanguage = useCallback((next: Language) => {
    setLanguageState(next)
    try { safeStorage()?.setItem(STORAGE_KEY, next) } catch { /* storage may be disabled; keep the in-memory choice */ }
    if (typeof window !== "undefined") window.dispatchEvent(new Event(changeEvent))
  }, [])
  const t = useCallback((key: MessageKey) => translate(language, key), [language])
  return { language, setLanguage, t }
}
