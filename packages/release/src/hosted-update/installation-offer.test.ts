import { Effect } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { describe, expect, it } from "vitest"
import { verifyInstallationOffer } from "./installation-offer"
import { signUpdateRelease, type ReleaseTarget } from "./release"

const keys = generateKeyPairSync("ed25519")
const trusted = new Map([["test", keys.publicKey]])
const target = { os: "darwin", arch: "arm64", package: "mac-zip" } as const
const offer = (version = "0.1.6") => signUpdateRelease({ version, bytes: 123, sha256: "a".repeat(64) }, target, keys.privateKey).pipe(
  Effect.map(release => ({ release, download: "https://github.com/magnitudedev/magnitude/releases/download/release/magnitude.zip" })))

describe("full installation release admission", () => {
  it("authenticates a fresh installation without inventing a previous version", async () => {
    const input = await Effect.runPromise(offer())
    expect(await Effect.runPromise(verifyInstallationOffer(input, target, "stable", trusted))).toEqual(input)
  })
  it.each<ReleaseTarget>([
    { ...target, arch: "x64" }, { ...target, package: "dmg" }, { os: "linux", arch: "arm64", package: "deb" },
  ])("rejects a release for another target", async selected => {
    const input = await Effect.runPromise(offer())
    expect(await Effect.runPromise(verifyInstallationOffer(input, selected, "stable", trusted).pipe(Effect.isFailure))).toBe(true)
  })
  it("rejects tampered content, an untrusted key and an unrelated download", async () => {
    const input = await Effect.runPromise(offer())
    for (const changed of [ { ...input, release: { ...input.release, bytes: 124 } },
      { ...input, download: "https://example.com/magnitude.zip" }, { ...input, execute: "/bin/sh" } ]) {
      expect(await Effect.runPromise(verifyInstallationOffer(changed, target, "stable", trusted).pipe(Effect.isFailure))).toBe(true)
    }
    expect(await Effect.runPromise(verifyInstallationOffer(input, target, "stable", new Map()).pipe(Effect.isFailure))).toBe(true)
  })
  it("applies the shared channel admission policy", async () => {
    const beta = await Effect.runPromise(offer("0.1.6-beta.1"))
    expect(await Effect.runPromise(verifyInstallationOffer(beta, target, "stable", trusted).pipe(Effect.isFailure))).toBe(true)
    expect(await Effect.runPromise(verifyInstallationOffer(beta, target, "beta", trusted))).toEqual(beta)
    expect(await Effect.runPromise(verifyInstallationOffer(beta, target, "alpha", trusted))).toEqual(beta)
  })
})
