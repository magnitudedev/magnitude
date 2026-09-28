import { ConfigProvider, Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import { resolveDesktopDistribution } from "./desktop-distribution"

const config = (entries: readonly (readonly [string, string])[]) => ConfigProvider.fromMap(new Map(entries))
const developerId = [
  ["MAGNITUDE_APPLE_DISTRIBUTION", "developer-id"],
  ["APPLE_SIGNING_IDENTITY", "Developer ID Application: Magnitude (ABCDEFGHIJ)"],
] as const
const artifactSigning = [["MAGNITUDE_WINDOWS_DISTRIBUTION", "artifact-signing"]] as const

describe("Desktop distribution", () => {
  it("ignores Apple and Windows signing configuration on Linux", async () => {
    const distribution = await Effect.runPromise(resolveDesktopDistribution("linux-x64-gnu").pipe(
      Effect.withConfigProvider(config([...developerId, ...artifactSigning])),
    ))
    expect(Option.isNone(distribution.appleTeam)).toBe(true)
    expect(Option.isNone(distribution.windowsPublisher)).toBe(true)
  })

  it("compiles the Team ID into a Developer ID macOS build and rejects a missing one", async () => {
    const distribution = await Effect.runPromise(resolveDesktopDistribution("darwin-arm64").pipe(
      Effect.withConfigProvider(config([...developerId, ["APPLE_TEAM_ID", "ABCDEFGHIJ"]])),
    ))
    expect(Option.getOrThrow(distribution.appleTeam)).toBe("ABCDEFGHIJ")
    await expect(Effect.runPromise(resolveDesktopDistribution("darwin-arm64").pipe(
      Effect.withConfigProvider(config(developerId)),
    ))).rejects.toThrow()
  })

  it("compiles the publisher into a signed Windows build and rejects a missing one", async () => {
    const distribution = await Effect.runPromise(resolveDesktopDistribution("windows-x64-msvc").pipe(
      Effect.withConfigProvider(config([...artifactSigning, ["MAGNITUDE_WINDOWS_PUBLISHER", "Magnitude"]])),
    ))
    expect(Option.getOrThrow(distribution.windowsPublisher)).toBe("Magnitude")
    await expect(Effect.runPromise(resolveDesktopDistribution("windows-x64-msvc").pipe(
      Effect.withConfigProvider(config(artifactSigning)),
    ))).rejects.toThrow()
  })

  it("carries no publisher identity in unsigned builds", async () => {
    for (const host of ["darwin-arm64", "windows-x64-msvc"] as const) {
      const distribution = await Effect.runPromise(resolveDesktopDistribution(host).pipe(Effect.withConfigProvider(config([]))))
      expect(Option.isNone(distribution.appleTeam) && Option.isNone(distribution.windowsPublisher)).toBe(true)
    }
  })
})
