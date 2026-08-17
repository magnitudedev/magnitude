import { beforeAll, describe, expect, test, vi } from "vitest"
import {
  routeSlashCommand,
  type CommandContext,
} from "@magnitudedev/client-common"
import { registerWebCommands } from "./register"

beforeAll(registerWebCommands)

describe("web local-model commands", () => {
  test.each(["models", "catalog", "hardware"] as const)(
    "routes /%s to its Settings tab",
    (tab) => {
      const openModelMenu = vi.fn()
      const context = {
        openModelMenu,
      } as unknown as CommandContext

      expect(routeSlashCommand(`/${tab}`, context)._tag).toBe("Handled")
      expect(openModelMenu).toHaveBeenCalledWith(tab)
    }
  )

  test("does not register cloud or usage commands", () => {
    const context = {} as CommandContext
    expect(routeSlashCommand("/cloud", context)._tag).toBe("Unhandled")
    expect(routeSlashCommand("/usage", context)._tag).toBe("Unhandled")
  })

  test("reopens the shared local-model onboarding flow", () => {
    const openSetup = vi.fn()
    const context = { openSetup } as unknown as CommandContext

    expect(routeSlashCommand("/setup", context)._tag).toBe("Handled")
    expect(openSetup).toHaveBeenCalledOnce()
  })
})
