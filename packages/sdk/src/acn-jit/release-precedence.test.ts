import { describe, expect, it } from "vitest"
import { compareReleaseVersions } from "./release-precedence"

describe("compareReleaseVersions", () => {
  it("implements SemVer core and prerelease ordering", () => {
    expect(compareReleaseVersions("2.0.0", "1.99.99")).toBe(1)
    expect(compareReleaseVersions("1.2.2", "1.2.3")).toBe(-1)
    expect(compareReleaseVersions("1.0.0", "1.0.0-rc.9")).toBe(1)
    expect(compareReleaseVersions("1.0.0-rc.10", "1.0.0-rc.2")).toBe(1)
    expect(compareReleaseVersions("1.0.0-alpha", "1.0.0-alpha.1")).toBe(-1)
    expect(compareReleaseVersions("1.0.0-1", "1.0.0-alpha")).toBe(-1)
    expect(
      compareReleaseVersions("999999999999999999999999.0.0", "999999999999999999999998.0.0"),
    ).toBe(1)
  })

  it("reuses exact identity and naturally orders build and arbitrary identities", () => {
    expect(compareReleaseVersions("1.2.3+build.1", "1.2.3+build.1")).toBe(0)
    expect(compareReleaseVersions("1.2.3+build.10", "1.2.3+build.2")).toBe(1)
    expect(compareReleaseVersions("dev-a10", "dev-a2")).toBe(1)
    expect(compareReleaseVersions("anything", "anything")).toBe(0)
  })

  it("orders Magnitude development identities by their generated timestamp", () => {
    expect(
      compareReleaseVersions(
        "0.0.1-alpha.22+dev.2c5b178.1784757574495",
        "0.0.1-alpha.22+dev.2c5b178.1784755698047",
      ),
    ).toBe(1)
    expect(
      compareReleaseVersions(
        "0.0.1-alpha.22+dev.2c5b178.1784755698047",
        "0.0.1-alpha.22+dev.2c5b178.1784757574495",
      ),
    ).toBe(-1)
    expect(
      compareReleaseVersions(
        "0.0.1-alpha.22+dev.newcommit.1784757574495",
        "0.0.1-alpha.22+dev.oldcommit.1784755698047",
      ),
    ).toBe(1)
  })

  it("orders a published release above a dev build of the same base", () => {
    expect(compareReleaseVersions("0.0.1-alpha.22", "0.0.1-alpha.22+dev.2c5b178.1")).toBe(1)
    expect(compareReleaseVersions("0.0.1-alpha.22+dev.2c5b178.1", "0.0.1-alpha.22")).toBe(-1)
  })

  it("deterministically orders malformed and non-Magnitude identities", () => {
    expect(compareReleaseVersions("release-10", "release-2")).toBe(1)
    expect(compareReleaseVersions("release-2", "release-10")).toBe(-1)
    expect(compareReleaseVersions("01", "1")).toBe(-1)
  })
})
