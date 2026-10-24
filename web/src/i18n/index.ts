import i18n from "i18next"
import { initReactI18next, useTranslation } from "react-i18next"
import { useCallback, useEffect } from "react"
import { en } from "./catalogs/en"
import { es } from "./catalogs/es"
import { zhCN } from "./catalogs/zh-CN"
import { fr } from "./catalogs/fr"
import { de } from "./catalogs/de"
import { ja } from "./catalogs/ja"
import { pt } from "./catalogs/pt"
import { it } from "./catalogs/it"
import { uk } from "./catalogs/uk"

export const LANGUAGES = ["en", "es", "zh-CN", "fr", "de", "ja", "pt", "it", "uk"] as const
export type Language = typeof LANGUAGES[number]
export type MessageKey = keyof typeof en
export const STORAGE_KEY = "magnitude.language"
const changeEvent = "magnitude-language-change"

const resources = {
  en: { translation: en },
  es: { translation: es },
  "zh-CN": { translation: zhCN },
  fr: { translation: fr },
  de: { translation: de },
  ja: { translation: ja },
  pt: { translation: pt },
  it: { translation: it },
  uk: { translation: uk },
} as const

export function isLanguage(value: unknown): value is Language {
  return typeof value === "string" && (LANGUAGES as readonly string[]).includes(value)
}

function safeStorage(): Pick<Storage, "getItem" | "setItem"> | undefined {
  try {
    return typeof localStorage === "undefined" ? undefined : localStorage
  } catch {
    return undefined
  }
}

export function readLanguage(storage: Pick<Storage, "getItem"> | undefined = safeStorage()): Language {
  try {
    const stored = storage?.getItem(STORAGE_KEY)
    return isLanguage(stored) ? stored : "en"
  } catch {
    return "en"
  }
}

void i18n.use(initReactI18next).init({
  resources,
  lng: readLanguage(),
  fallbackLng: "en",
  supportedLngs: [...LANGUAGES],
  defaultNS: "translation",
  ns: ["translation"],
  keySeparator: false,
  interpolation: { escapeValue: false },
  initImmediate: false,
})

function applyDocumentLanguage(language: Language): void {
  if (typeof document === "undefined") return
  document.documentElement.lang = language
  document.documentElement.dir = "ltr"
}

if (typeof window !== "undefined") {
  applyDocumentLanguage(readLanguage())
  window.addEventListener("storage", event => {
    if (event.key === STORAGE_KEY) void i18n.changeLanguage(readLanguage())
  })
  window.addEventListener(changeEvent, () => {
    void i18n.changeLanguage(readLanguage())
  })
  i18n.on("languageChanged", language => {
    if (isLanguage(language)) applyDocumentLanguage(language)
  })
}

/** Compatibility wrapper for components being migrated from the former hand-written layer. */
export function useI18n() {
  const { t, i18n: instance } = useTranslation()
  const language: Language = isLanguage(instance.resolvedLanguage) ? instance.resolvedLanguage
    : isLanguage(instance.language) ? instance.language : "en"

  useEffect(() => applyDocumentLanguage(language), [language])

  const setLanguage = useCallback((next: Language) => {
    try { safeStorage()?.setItem(STORAGE_KEY, next) } catch { /* Keep the in-memory choice when storage is unavailable. */ }
    void instance.changeLanguage(next)
    if (typeof window !== "undefined") window.dispatchEvent(new Event(changeEvent))
  }, [instance])

  const translate = useCallback((key: MessageKey, options?: { count?: number }) => t(key, options), [t])
  return { language, setLanguage, t: translate }
}

const englishMessageKeys = new Map<string, MessageKey>()
for (const [key, value] of Object.entries(en) as Array<[MessageKey, string]>) {
  if (!englishMessageKeys.has(value)) englishMessageKeys.set(value, key)
}

/** Translate exact known UI messages from service/host layers; preserve unknown diagnostics verbatim. */
export function translateKnownMessage(message: string, language: Language = readLanguage()): string {
  const key = englishMessageKeys.get(message)
  return key ? i18n.t(key, { lng: language }) : message
}

export { i18n }
