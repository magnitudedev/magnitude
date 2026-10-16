import { describe, expect, test } from "vitest"
import { isLanguage, readLanguage, STORAGE_KEY, translate } from "./index"
import { en } from "./catalogs/en"
import { es } from "./catalogs/es"
import { zhCN } from "./catalogs/zh-CN"
import { hi } from "./catalogs/hi"
import { ur } from "./catalogs/ur"
import { bn } from "./catalogs/bn"

describe("i18n catalogs", () => {
  test("supports English, Spanish, Simplified Chinese, Hindi, Urdu and Bengali", () => {
    expect(translate("en", "settings")).toBe("Settings")
    expect(translate("es", "settings")).toBe("Ajustes")
    expect(translate("zh-CN", "settings")).toBe("设置")
    expect(translate("hi", "settings")).toBe("सेटिंग्स")
    expect(translate("ur", "settings")).toBe("ترتیبات")
    expect(translate("bn", "settings")).toBe("সেটিংস")
  })

  test("rejects unknown locale values", () => {
    expect(isLanguage("es")).toBe(true)
    expect(isLanguage("zh-CN")).toBe(true)
    expect(isLanguage("hi")).toBe(true)
    expect(isLanguage("ur")).toBe(true)
    expect(isLanguage("bn")).toBe(true)
    expect(isLanguage("fr")).toBe(false)
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

  test("catalogs share the same message keys and every locale has a translation", () => {
    expect(Object.keys(es).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(zhCN).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(hi).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(ur).sort()).toEqual(Object.keys(en).sort())
    expect(Object.keys(bn).sort()).toEqual(Object.keys(en).sort())
    for (const key of Object.keys(en) as Array<keyof typeof en>) {
      expect(en[key], `English: ${key}`).toBeTruthy()
      expect(es[key], `Spanish: ${key}`).toBeTruthy()
      expect(zhCN[key], `Chinese: ${key}`).toBeTruthy()
      expect(hi[key], `Hindi: ${key}`).toBeTruthy()
      expect(ur[key], `Urdu: ${key}`).toBeTruthy()
      expect(bn[key], `Bengali: ${key}`).toBeTruthy()
    }
  })
})
