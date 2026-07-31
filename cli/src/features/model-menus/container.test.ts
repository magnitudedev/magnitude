import { describe, expect, test } from "vitest"
import type { KeyEvent } from "@opentui/core"
import { resolveRootNavigationDirection } from "./container"

const key = (
  name: string,
  overrides: Partial<Pick<KeyEvent, "ctrl" | "meta" | "option" | "shift">> = {},
): Pick<KeyEvent, "name" | "ctrl" | "meta" | "option" | "shift"> => ({
  name,
  ctrl: false,
  meta: false,
  option: false,
  shift: false,
  ...overrides,
})

describe("model menu root navigation", () => {
  test("resolves lateral navigation without depending on the nested view", () => {
    expect(resolveRootNavigationDirection(key("left"))).toBe(-1)
    expect(resolveRootNavigationDirection(key("right"))).toBe(1)
  })

  test("resolves forward and reverse tab navigation", () => {
    expect(resolveRootNavigationDirection(key("tab"))).toBe(1)
    expect(resolveRootNavigationDirection(key("tab", { shift: true }))).toBe(-1)
  })

  test("leaves modified navigation keys unhandled", () => {
    expect(resolveRootNavigationDirection(key("left", { ctrl: true }))).toBeNull()
    expect(resolveRootNavigationDirection(key("right", { meta: true }))).toBeNull()
    expect(resolveRootNavigationDirection(key("tab", { option: true }))).toBeNull()
  })

  test("leaves unrelated keys to the active view", () => {
    expect(resolveRootNavigationDirection(key("escape"))).toBeNull()
    expect(resolveRootNavigationDirection(key("up"))).toBeNull()
  })
})
