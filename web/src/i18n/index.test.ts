import { describe, expect, test } from "vitest"
import { isLanguage, readLanguage, STORAGE_KEY, translate } from "./index"

describe("i18n catalogs", () => {
  test("supports English, Spanish and Simplified Chinese", () => {
    expect(translate("en", "settings")).toBe("Settings")
    expect(translate("es", "settings")).toBe("Ajustes")
    expect(translate("zh-CN", "settings")).toBe("设置")
  })

  test("rejects unknown locale values", () => {
    expect(isLanguage("es")).toBe(true)
    expect(isLanguage("zh-CN")).toBe(true)
    expect(isLanguage("fr")).toBe(false)
    expect(isLanguage(null)).toBe(false)
  })

  test("uses English when no valid stored language exists", () => {
    expect(readLanguage({ getItem: () => null })).toBe("en")
    expect(readLanguage({ getItem: () => "xx" })).toBe("en")
    expect(readLanguage({ getItem: key => key === STORAGE_KEY ? "es" : null })).toBe("es")
  })

  test("catalogs share the same message keys", () => {
    for (const key of ["settings", "general", "about", "thisBrowser", "language", "languageHint", "english", "spanish", "chineseSimplified"] as const) {
      expect(translate("en", key)).toBeTruthy()
      expect(translate("es", key)).toBeTruthy()
      expect(translate("zh-CN", key)).toBeTruthy()
    }
  })
})
