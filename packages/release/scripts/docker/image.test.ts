import { Effect, Exit } from "effect"
import { describe, expect, it } from "vitest"
import { debArtifactId, imageTags, parseDockerArchitecture } from "./image"

describe("Docker image", () => {
  it("tags a stable release with its version and latest", () => {
    expect(Effect.runSync(imageTags("magnitudedev/magnitude", "0.2.14"))).toEqual([
      "magnitudedev/magnitude:0.2.14",
      "magnitudedev/magnitude:latest",
    ])
  })

  it("tags a prerelease with its version and its channel, never latest", () => {
    expect(Effect.runSync(imageTags("magnitudedev/magnitude", "0.3.0-beta.2"))).toEqual([
      "magnitudedev/magnitude:0.3.0-beta.2",
      "magnitudedev/magnitude:beta",
    ])
    expect(Effect.runSync(imageTags("magnitudedev/magnitude", "0.3.0-alpha.0"))).toEqual([
      "magnitudedev/magnitude:0.3.0-alpha.0",
      "magnitudedev/magnitude:alpha",
    ])
  })

  it("refuses versions that are not releases or not valid tags", () => {
    for (const version of ["latest", "0.2", "0.2.14+build.1", ""]) {
      expect(Exit.isFailure(Effect.runSyncExit(imageTags("magnitudedev/magnitude", version)))).toBe(true)
    }
  })

  it("maps Docker architectures to the release's Debian packages", () => {
    expect(debArtifactId(Effect.runSync(parseDockerArchitecture("amd64")))).toBe("desktop-linux-x64-gnu-deb")
    expect(debArtifactId(Effect.runSync(parseDockerArchitecture("arm64")))).toBe("desktop-linux-arm64-gnu-deb")
    expect(Exit.isFailure(Effect.runSyncExit(parseDockerArchitecture("x86_64")))).toBe(true)
  })
})
