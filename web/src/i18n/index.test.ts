import { describe, expect, test } from "vitest"
import { isLanguage, readLanguage, STORAGE_KEY, translate, translateKnownMessage } from "./index"
import { en } from "./catalogs/en"
import { es } from "./catalogs/es"
import { zhCN } from "./catalogs/zh-CN"
import { hi } from "./catalogs/hi"
import { ur } from "./catalogs/ur"
import { bn } from "./catalogs/bn"
import { fr } from "./catalogs/fr"
import { de } from "./catalogs/de"
import { ja } from "./catalogs/ja"
import { pt } from "./catalogs/pt"
import { it } from "./catalogs/it"
import { uk } from "./catalogs/uk"

describe("i18n catalogs", () => {
  test("supports English, Spanish, Simplified Chinese, Hindi, Urdu, Bengali, French, German, Japanese, Portuguese, Italian and Ukrainian", () => {
    expect(translate("en", "settings")).toBe("Settings")
    expect(translate("es", "settings")).toBe("Ajustes")
    expect(translate("zh-CN", "settings")).toBe("设置")
    expect(translate("hi", "settings")).toBe("सेटिंग्स")
    expect(translate("ur", "settings")).toBe("ترتیبات")
    expect(translate("bn", "settings")).toBe("সেটিংস")
    expect(translate("fr", "settings")).toBe("Paramètres")
    expect(translate("de", "settings")).toBe("Einstellungen")
    expect(translate("ja", "settings")).toBe("設定")
    expect(translate("pt", "settings")).toBe("Configurações")
    expect(translate("it", "settings")).toBe("Impostazioni")
    expect(translate("uk", "settings")).toBe("Налаштування")
  })

  test("translates exact known messages but preserves unknown diagnostics", () => {
    expect(translateKnownMessage("Couldn’t load the model catalog", "es")).toBe("No se pudo cargar el catálogo de modelos")
    expect(translateKnownMessage("unrecognized diagnostic from the service", "es")).toBe("unrecognized diagnostic from the service")
  })

  test("rejects unknown locale values", () => {
    expect(isLanguage("es")).toBe(true)
    expect(isLanguage("zh-CN")).toBe(true)
    expect(isLanguage("hi")).toBe(true)
    expect(isLanguage("ur")).toBe(true)
    expect(isLanguage("bn")).toBe(true)
    expect(isLanguage("fr")).toBe(true)
    expect(isLanguage("de")).toBe(true)
    expect(isLanguage("ja")).toBe(true)
    expect(isLanguage("pt")).toBe(true)
    expect(isLanguage("it")).toBe(true)
    expect(isLanguage("uk")).toBe(true)
    expect(isLanguage("xx")).toBe(false)
    expect(isLanguage(null)).toBe(false)
  })

  test("uses English when no valid stored language exists", () => {
    expect(readLanguage({ getItem: () => null })).toBe("en")
    expect(readLanguage({ getItem: () => "xx" })).toBe("en")
    expect(readLanguage({ getItem: key => key === STORAGE_KEY ? "es" : null })).toBe("es")
  })

  test("falls back to English if storage access throws", () => {
    expect(readLanguage({ getItem: () => { throw new Error("storage blocked") } })).toBe("en")
  })

  test("preserves interpolation placeholders in every locale", () => {
    const placeholders = (message: string) => [...message.matchAll(/\{[A-Za-z0-9_]+\}/g)].map(match => match[0]).sort()
    for (const key of Object.keys(en) as Array<keyof typeof en>) {
      expect(placeholders(es[key]), `Spanish placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(zhCN[key]), `Chinese placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(hi[key]), `Hindi placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(ur[key]), `Urdu placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(bn[key]), `Bengali placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(fr[key]), `French placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(de[key]), `German placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(ja[key]), `Japanese placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(pt[key]), `Portuguese placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(it[key]), `Italian placeholders: ${key}`).toEqual(placeholders(en[key]))
      expect(placeholders(uk[key]), `Ukrainian placeholders: ${key}`).toEqual(placeholders(en[key]))
    }
  })

  test("catalogs share the same message keys and every locale has a translation", () => {
    expect(Object.keys(es).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(zhCN).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(hi).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(ur).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(bn).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(fr).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(de).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(ja).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(pt).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(it).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(uk).sort()).toEqual(Object.keys(en).sort())
    for (const key of Object.keys(en) as Array<keyof typeof en>) {
      expect(en[key], `English: ${key}`).toBeTruthy()
      expect(es[key], `Spanish: ${key}`).toBeTruthy()
      expect(zhCN[key], `Chinese: ${key}`).toBeTruthy()
      expect(hi[key], `Hindi: ${key}`).toBeTruthy()
      expect(ur[key], `Urdu: ${key}`).toBeTruthy()
      expect(bn[key], `Bengali: ${key}`).toBeTruthy()
      expect(fr[key], `French: ${key}`).toBeTruthy()
      expect(de[key], `German: ${key}`).toBeTruthy()
      expect(ja[key], `Japanese: ${key}`).toBeTruthy()
      expect(pt[key], `Portuguese: ${key}`).toBeTruthy()
      expect(it[key], `Italian: ${key}`).toBeTruthy()
      expect(uk[key], `Ukrainian: ${key}`).toBeTruthy()
    }
  })
})
