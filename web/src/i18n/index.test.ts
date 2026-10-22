import { describe, expect, test } from "vitest"
import { i18n, isLanguage, LANGUAGES, readLanguage, STORAGE_KEY, translateKnownMessage } from "./index"
import { en } from "./catalogs/en"
import { es } from "./catalogs/es"
import { zhCN } from "./catalogs/zh-CN"
import { fr } from "./catalogs/fr"
import { de } from "./catalogs/de"
import { ja } from "./catalogs/ja"
import { pt } from "./catalogs/pt"
import { it } from "./catalogs/it"
import { uk } from "./catalogs/uk"

const catalogs = { en, es, "zh-CN": zhCN, fr, de, ja, pt, it, uk }

describe("react-i18next locale resources", () => {
  test("exposes exactly the nine requested LTR locales", () => {
    expect(LANGUAGES).toEqual(["en", "es", "zh-CN", "fr", "de", "ja", "pt", "it", "uk"])
    for (const language of LANGUAGES) expect(isLanguage(language)).toBe(true)
    for (const excluded of ["ur", "zh-TW", "id", "ko", "vi", "xx", null]) expect(isLanguage(excluded)).toBe(false)
  })

  test("translates representative labels through i18next", () => {
    const examples = [
      ["en", "Settings"], ["es", "Ajustes"], ["zh-CN", "设置"],
      ["fr", "Paramètres"], ["de", "Einstellungen"], ["ja", "設定"],
      ["pt", "Configurações"], ["it", "Impostazioni"], ["uk", "Налаштування"],
    ] as const
    for (const [language, expected] of examples) {
      expect(i18n.t("settings", { lng: language })).toBe(expected)
    }
  })

  test("translates navigation and settings chrome through exact known labels", () => {
    const cases = [
      ["General", "es", "General"],
      ["General", "zh-CN", "常规"],
      ["General", "fr", "Général"],
      ["Theme", "es", "Tema"],
      ["Theme", "zh-CN", "主题"],
      ["Theme", "uk", "Тема"],
      ["Open navigation", "es", "Abrir navegación"],
      ["Open navigation", "zh-CN", "打开导航"],
      ["Open navigation", "ja", "ナビゲーションを開く"],
    ] as const
    for (const [english, language, expected] of cases) {
      expect(translateKnownMessage(english, language)).toBe(expected)
    }
  })

  test("translates known service messages and preserves unknown diagnostics", () => {
    expect(translateKnownMessage("Couldn’t load the model catalog", "es")).toBe("No se pudo cargar el catálogo de modelos")
    const diagnostic = "unrecognized diagnostic from the service"
    expect(translateKnownMessage(diagnostic, "es")).toBe(diagnostic)
  })

  test("uses English when no valid stored language exists and handles blocked storage", () => {
    expect(readLanguage({ getItem: () => null })).toBe("en")
    expect(readLanguage({ getItem: () => "xx" })).toBe("en")
    expect(readLanguage({ getItem: key => key === STORAGE_KEY ? "es" : null })).toBe("es")
    expect(readLanguage({ getItem: () => { throw new Error("storage blocked") } })).toBe("en")
  })

  test("all nine catalogs have identical non-empty message keys", () => {
    const expectedKeys = Object.keys(en).sort()
    for (const [language, catalog] of Object.entries(catalogs)) {
      expect(Object.keys(catalog).sort(), language).toEqual(expectedKeys)
      for (const key of expectedKeys as Array<keyof typeof en>) {
        expect(catalog[key], `${language} translation: ${key}`).toBeTruthy()
      }
    }
  })

  test("all catalogs preserve i18next interpolation variables", () => {
    const placeholders = (message: string) => [...message.matchAll(/\{\{[A-Za-z0-9_]+\}\}/g)].map(match => match[0]).sort()
    for (const [language, catalog] of Object.entries(catalogs)) {
      for (const key of Object.keys(en) as Array<keyof typeof en>) {
        expect(placeholders(catalog[key]), `${language} placeholders: ${key}`).toEqual(placeholders(en[key]))
      }
    }
  })

  test("pluralizes the model search placeholder with each locale's rules", () => {
    const cases = [
      ["en", 1, "Search 1 model…"],
      ["en", 3, "Search 3 models…"],
      ["en", 39, "Search 39 models…"],
      ["es", 1, "Buscar 1 modelo…"],
      ["es", 3, "Buscar 3 modelos…"],
      ["uk", 1, "Шукати 1 модель…"],
      ["uk", 3, "Шукати 3 моделі…"],
      ["uk", 5, "Шукати 5 моделей…"],
      ["uk", 39, "Шукати 39 моделей…"],
      ["zh-CN", 3, "搜索 3 个模型…"],
    ] as const
    for (const [language, count, expected] of cases) {
      expect(i18n.t("searchModelsPlaceholder", { lng: language, count })).toBe(expected)
    }
  })

  test("does not configure any RTL locale", () => {
    expect(i18n.options.supportedLngs).toEqual(expect.arrayContaining([...LANGUAGES]))
    expect(LANGUAGES).not.toContain("ur" as never)
  })
})
