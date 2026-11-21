import { Effect, Exit } from "effect"
import { describe, expect, it } from "vitest"
import { cliOnlyArtifactId, imageTags, parseDockerArchitecture } from "./image"

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

  it("maps Docker architectures to the release's CLI-only payloads", () => {
    expect(cliOnlyArtifactId(Effect.runSync(parseDockerArchitecture("amd64")))).toBe("cli-only-linux-x64-gnu")
    expect(cliOnlyArtifactId(Effect.runSync(parseDockerArchitecture("arm64")))).toBe("cli-only-linux-arm64-gnu")
    expect(Exit.isFailure(Effect.runSyncExit(parseDockerArchitecture("x86_64")))).toBe(true)
  })
})
