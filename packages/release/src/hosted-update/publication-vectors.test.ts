import { createPrivateKey, createPublicKey } from "node:crypto"
import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { signUpdateManifest, UpdateManifest } from "./manifest"
import { verifyUpdateRelease } from "./release"
import vectors from "./publication-vectors.json"

// Public all-zero seed: deterministic protocol fixture, never a production key.
const privateKey = createPrivateKey({ key: Buffer.concat([Buffer.from("302e020100300506032b657004220420", "hex"), Buffer.alloc(32)]), format: "der", type: "pkcs8" })
const trust = new Map([["fixture", createPublicKey(vectors.publicKey)]])
describe("website publication contract", () => {
  for (const vector of vectors.publications) {
    it(`preserves ${vector.manifest.artifact.target.os}/${vector.manifest.artifact.target.package} signed bytes`, async () => {
      const manifest = Schema.decodeUnknownSync(UpdateManifest)(vector.manifest)
      expect(await Effect.runPromise(signUpdateManifest(manifest, privateKey))).toEqual(vector)
      expect(await Effect.runPromise(verifyUpdateRelease(vector.release, manifest.artifact.target, trust))).toEqual(vector.release)
    })
  }
})
