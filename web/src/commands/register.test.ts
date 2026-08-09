import { beforeAll, describe, expect, test, vi } from "vitest"
import { routeSlashCommand, type CommandContext } from "@magnitudedev/client-common"
import { registerWebCommands } from "./register"

beforeAll(registerWebCommands)

describe("web local-model commands", () => {
  test.each(["models", "catalog", "hardware"] as const)("routes /%s to its Model Center tab", (tab) => {
    const openModelMenu = vi.fn()
    const context = {
      openModelMenu,
    } as unknown as CommandContext

    expect(routeSlashCommand(`/${tab}`, context)).toBe(true)
    expect(openModelMenu).toHaveBeenCalledWith(tab)
  })

  test("does not register cloud or usage commands", () => {
    const context = {} as CommandContext
    expect(routeSlashCommand("/cloud", context)).toBe(false)
    expect(routeSlashCommand("/usage", context)).toBe(false)
  })
})
