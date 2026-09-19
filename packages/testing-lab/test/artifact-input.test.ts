import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import releasePlan from "../../release/release-plan.json"
import { snapshotArtifacts } from "../src/artifact-input"
import { sha256 } from "../src/snapshot"

const json = Schema.encodeSync(Schema.parseJson(Schema.Unknown))
const decode = Schema.decodeUnknownSync(Schema.parseJson(Schema.Unknown))
const payload = "unpublished package"
const manifest = { schemaVersion: 2, version: "0.1.3", acnRevision: 1, rpc: releasePlan.rpc, plugins: [],
  tag: "@magnitudedev/cli@0.1.3", sourceCommit: "a".repeat(40), artifacts: [{ id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64",
    filename: "Magnitude.dmg", bytes: Buffer.byteLength(payload), sha256: sha256(payload) }] }

test("freezes an unpublished package and rejects missing, modified, wrong-length and escaping inputs", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-artifact-input-" })
  const path = join(root, "release.json")
  const file = join(root, "Magnitude.dmg")
  const objects = join(root, "objects")
  yield* fs.writeFileString(path, json(manifest))
  expect((yield* snapshotArtifacts(path, objects).pipe(Effect.either))._tag).toBe("Left")
  yield* fs.writeFileString(file, payload)
  const snapshot = yield* snapshotArtifacts(path, objects)
  expect(snapshot.digests).toEqual([sha256(payload)])
  expect(snapshot.digest).toBe(sha256(snapshot.json))
  expect(decode(snapshot.json)).toMatchObject({ release: manifest })
  yield* fs.writeFileString(file, "altered package")
  expect(yield* fs.readFileString(join(objects, sha256(payload)))).toBe(payload)
  expect((yield* snapshotArtifacts(path, objects).pipe(Effect.either))._tag).toBe("Left")
  expect(yield* fs.readFileString(join(objects, sha256(payload)))).toBe(payload)
  yield* fs.writeFileString(file, payload)
  yield* fs.writeFileString(path, json({ ...manifest, artifacts: [{ ...manifest.artifacts[0], bytes: 1 }] }))
  expect((yield* snapshotArtifacts(path, objects).pipe(Effect.either))._tag).toBe("Left")
  yield* fs.writeFileString(path, json({ ...manifest, artifacts: [{ ...manifest.artifacts[0], filename: "../outside.dmg" }] }))
  expect((yield* snapshotArtifacts(path, objects).pipe(Effect.either))._tag).toBe("Left")
})).pipe(Effect.provide(BunContext.layer))))
