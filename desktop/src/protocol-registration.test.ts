import { describe, expect, it } from "vitest"
import { Effect } from "effect"
import { resolve } from "node:path"
import { registerDevelopmentProtocolClient } from "./protocol-registration"

const register = (platform: NodeJS.Platform, environment: NodeJS.ProcessEnv) => {
  const calls: unknown[][] = []
  const app = { setAsDefaultProtocolClient: (...args: unknown[]) => { calls.push(args); return true } }
  Effect.runSync(registerDevelopmentProtocolClient({ app, platform, environment, argv: ["/electron", "app"], execPath: "/electron" }))
  return calls
}

describe("development protocol registration", () => {
  it("leaves the installed handler alone unless asked", () => {
    expect(register("darwin", {})).toEqual([])
    expect(register("win32", { MAGNITUDE_DESKTOP_REGISTER_PROTOCOL: "0" })).toEqual([])
  })
  it("registers the running build on request, relaunching the app directory on Windows", () => {
    expect(register("darwin", { MAGNITUDE_DESKTOP_REGISTER_PROTOCOL: "1" })).toEqual([["magnitude"]])
    expect(register("win32", { MAGNITUDE_DESKTOP_REGISTER_PROTOCOL: "1" })).toEqual([["magnitude", "/electron", [resolve("app")]]])
  })
  it("never registers on Linux, where the handler is the installed .desktop entry", () => {
    expect(register("linux", { MAGNITUDE_DESKTOP_REGISTER_PROTOCOL: "1" })).toEqual([])
  })
})
