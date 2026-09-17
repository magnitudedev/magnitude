import { describe, expect, it } from "vitest"
import {
  admittedChannels,
  isNewerVersion,
  newestFirst,
  releaseChannelOf,
} from "./release-channels"

describe("release channels", () => {
  it("derives a channel from the prerelease identifier", () => {
    expect(releaseChannelOf("1.2.3")).toBe("stable")
    expect(releaseChannelOf("1.2.3-alpha.4")).toBe("alpha")
    expect(releaseChannelOf("1.2.3-beta.1")).toBe("beta")
    expect(releaseChannelOf("1.2.3-rc.1")).toBe("unknown")
    expect(releaseChannelOf("not-a-version")).toBe("stable")
  })

  it("admits candidates by the client's channel, unknown nowhere", () => {
    expect(admittedChannels("stable")).toEqual(new Set(["stable"]))
    expect(admittedChannels("beta")).toEqual(new Set(["stable", "beta"]))
    expect(admittedChannels("alpha")).toEqual(new Set(["stable", "beta", "alpha"]))
    expect(admittedChannels("unknown")).toEqual(new Set(["stable"]))
    expect(admittedChannels("stable").has("unknown")).toBe(false)
  })

  it("orders across channels by semver, prereleases below their release", () => {
    expect(isNewerVersion("1.0.0", "1.0.0-beta.2")).toBe(true)
    expect(isNewerVersion("1.0.0-beta.2", "1.0.0-alpha.9")).toBe(true)
    expect(isNewerVersion("1.1.0-alpha.1", "1.0.0")).toBe(true)
    expect(isNewerVersion("not-a-version", "1.0.0")).toBe(false)
    expect(newestFirst(["1.0.0", "1.1.0-beta.2", "1.2.0-alpha.1"]))
      .toEqual(["1.2.0-alpha.1", "1.1.0-beta.2", "1.0.0"])
  })
})
