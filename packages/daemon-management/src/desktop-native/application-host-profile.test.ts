import { homedir } from "node:os"
import { join } from "node:path"
import { Option } from "effect"
import { afterEach, describe, expect, it, vi } from "vitest"
import { makeDesktopApplicationHost } from "./application-host"

afterEach(() => vi.unstubAllEnvs())

describe("desktop host profile selection", () => {
  it("keeps ordinary packaged commands on the production profile", () => {
    vi.stubEnv("MAGNITUDE_DEV_DATA_DIR", undefined)
    vi.stubEnv("MAGNITUDE_DEV_PORT", "11219")
    const host = makeDesktopApplicationHost(Option.none())
    expect(host.desktopIsolatedProfile).toBe(false)
    expect(host.desktopDataDirectory).toBe(join(homedir(), ".magnitude"))
    expect(host.desktopServiceOrigin).toBe("http://127.0.0.1:10100")
  })

  it("applies an explicit private profile to a packaged CLI", () => {
    const root = join(homedir(), "private-test-profile")
    vi.stubEnv("MAGNITUDE_DEV_DATA_DIR", root)
    vi.stubEnv("MAGNITUDE_DEV_PORT", "11219")
    const host = makeDesktopApplicationHost(Option.none())
    expect(host.desktopIsolatedProfile).toBe(true)
    expect(host.desktopDataDirectory).toBe(root)
    expect(host.desktopServiceOrigin).toBe("http://127.0.0.1:11219")
  })

  it("uses the same private default port as desktop without a port override", () => {
    vi.stubEnv("MAGNITUDE_DEV_DATA_DIR", join(homedir(), "private-test-profile"))
    vi.stubEnv("MAGNITUDE_DEV_PORT", undefined)
    expect(makeDesktopApplicationHost(Option.none()).desktopServiceOrigin).toBe("http://127.0.0.1:11101")
  })

  it("retains source-checkout development defaults", () => {
    vi.stubEnv("MAGNITUDE_DEV_DATA_DIR", undefined)
    vi.stubEnv("MAGNITUDE_DEV_PORT", undefined)
    const host = makeDesktopApplicationHost(Option.some(join(homedir(), "checkout")))
    expect(host.desktopIsolatedProfile).toBe(true)
    expect(host.desktopDataDirectory).toBe(join(homedir(), ".magnitude-desktop-dev"))
    expect(host.desktopServiceOrigin).toBe("http://127.0.0.1:11101")
  })
})
