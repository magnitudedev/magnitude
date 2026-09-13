import { generateKeyPairSync } from "node:crypto"
import { NodeContext } from "@effect/platform-node"
import { FetchHttpClient } from "@effect/platform"
import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { publishHostedRelease, ReleasePublicationStore } from "./publication"
import { PublisherKeyId, UpdateManifest } from "./manifest"

const publisher = generateKeyPairSync("ed25519"), keyId = PublisherKeyId.make("test")
const artifact = (pkg: "dmg" | "mac-zip", version = "2.0.0") => ({ file: `/missing-magnitude-candidate/${pkg}`, manifest: Schema.decodeUnknownSync(UpdateManifest)({
  protocol: 1, version, commit: "a".repeat(40), artifact: { id: pkg, target: { os: "darwin", arch: "arm64", package: pkg }, path: `releases/${version}/mac.${pkg}`, bytes: 100, sha256: "a".repeat(64) },
}) })
const options = { artifacts: [artifact("dmg"), artifact("mac-zip")], keyId, privateKey: publisher.privateKey, token: "test", storageOrigin: "https://downloads.magnitude.dev" }
const run = async (artifacts: typeof options.artifacts) => {
  let promoted = false
  const result = await Effect.runPromise(publishHostedRelease({ ...options, artifacts }).pipe(
    Effect.provideService(ReleasePublicationStore, { promote: () => Effect.sync(() => { promoted = true }) }),
    Effect.provide([NodeContext.layer, FetchHttpClient.layer]), Effect.either,
  ))
  expect(promoted).toBe(false)
  if (result._tag !== "Left") throw new Error("Invalid publication succeeded")
  return result.left
}
describe("release publication admission", () => {
  it("rejects mixed cohorts before any artifact access or channel promotion", async () => {
    expect(await run([artifact("dmg"), artifact("mac-zip", "3.0.0")])).toMatchObject({ _tag: "ReleasePublicationFailed", stage: "batch" })
  })
  it("does not promote a coherent release whose local bytes cannot be verified", async () => {
    expect(await run(options.artifacts)).toMatchObject({ _tag: "ArtifactPublicationFailed", stage: "local-verification" })
  })
})
