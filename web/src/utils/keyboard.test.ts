import { describe, expect, test } from "vitest"
import { toGenericKeyEvent } from "./keyboard"

const key = (name: string, init: KeyboardEventInit = {}) =>
  toGenericKeyEvent({
    key: name,
    ctrlKey: init.ctrlKey ?? false,
    metaKey: init.metaKey ?? false,
    shiftKey: init.shiftKey ?? false,
    altKey: init.altKey ?? false,
    defaultPrevented: false,
  } as KeyboardEvent)

describe("DOM keyboard normalization", () => {
  test.each([
    ["Enter", "enter"],
    ["Escape", "escape"],
    ["Tab", "tab"],
    ["ArrowUp", "up"],
    ["ArrowDown", "down"],
  ])("maps %s to the shared key name %s", (domName, sharedName) => {
    expect(key(domName).name).toBe(sharedName)
  })

  test("preserves modifier state", () => {
    expect(key("Enter", { ctrlKey: true, metaKey: true, shiftKey: true, altKey: true })).toMatchObject({
      name: "enter",
      ctrl: true,
      meta: true,
      shift: true,
      option: true,
    })
  })
})
