import { describe, expect, it } from "vitest"
import { generateCssVars } from "./generate-css-vars"

describe("web semantic color variables", () => {
  it("provides complete light and dark semantic palettes", () => {
    const dark = generateCssVars("dark")
    const light = generateCssVars("light")
    expect(Object.keys(light).sort()).toEqual(Object.keys(dark).sort())
    expect(light["--bg-base"]).not.toBe(dark["--bg-base"])
    expect(light["--fg-primary"]).not.toBe(dark["--fg-primary"])
    expect(light["--syntax-keyword"]).not.toBe(dark["--syntax-keyword"])
    expect(light["--border-default"]).not.toBe(dark["--border-default"])
  })

  it("retains dark mode as the compatibility default for static callers", () => {
    expect(generateCssVars()).toEqual(generateCssVars("dark"))
  })
})
