import { FetchHttpClient } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { findReleaseUpdate } from "./discovery"

class CandidateUnavailable extends Schema.TaggedError<CandidateUnavailable>()("CandidateUnavailable", {}) {}

describe("shared release discovery", () => {
  it("returns the exact verified payload, skips unavailable candidates and deduplicates tags", async () => {
    const server = Bun.serve({ port: 0, fetch: () => Response.json({ latest: "2.0.0", beta: "2.0.0", alpha: "1.5.0-alpha.1" }) })
    const attempts: string[] = []
    const payload = { archive: "verified-desktop.zip", sha256: "a".repeat(64) }
    try {
      const result = await Effect.runPromise(findReleaseUpdate({
        currentVersion: "1.0.0-alpha.1", registryUrl: `${server.url}registry`,
        verify: version => Effect.gen(function* () {
          attempts.push(version)
          if (version === "2.0.0") return yield* new CandidateUnavailable({})
          return payload
        }),
      }).pipe(Effect.provide(FetchHttpClient.layer)))
      expect(attempts).toEqual(["2.0.0", "1.5.0-alpha.1"])
      expect(Option.getOrThrow(result)).toBe(payload)
    } finally { server.stop(true) }
  })

  it("rejects oversized registry responses before any artifact verification", async () => {
    const server = Bun.serve({ port: 0, fetch: () => new Response(" ".repeat(65537)) })
    let verified = false
    try {
      const error = await Effect.runPromise(findReleaseUpdate({
        currentVersion: "1.0.0", registryUrl: `${server.url}registry`,
        verify: () => Effect.sync(() => { verified = true }),
      }).pipe(Effect.flip, Effect.provide(FetchHttpClient.layer)))
      expect(error).toMatchObject({ _tag: "UpdateDiscoveryFailed", stage: "registry", reason: "npm registry response exceeds its size bound" })
      expect(verified).toBe(false)
    } finally { server.stop(true) }
  })
})
