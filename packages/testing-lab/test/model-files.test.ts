import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { declaredModelFiles, verifyModelFiles } from "../src/model-files"
import { sha256 } from "../src/snapshot"

const bytes = new TextEncoder().encode("exact model bytes"), companion = new TextEncoder().encode("projector")
const source = { _tag: "HuggingFace", repository: "owner/model", revision: "a".repeat(40) }
const file = { path: "weights.gguf", sizeBytes: bytes.length, sha256: sha256(bytes) }
const entry = { modelId: "model", variantId: "gguf:q4", target: { package: { source, files: [file,
  { path: "nested/projector.gguf", sizeBytes: companion.length, sha256: sha256(companion) }] } } }
const bundle = (inputs: unknown) => {
  const manifest = Buffer.from(JSON.stringify({ plannerInputs: inputs })), header = Buffer.alloc(16)
  header.write("MAGPLAN3"); header.writeBigUInt64LE(BigInt(manifest.length), 8)
  return Buffer.concat([header, manifest, Buffer.alloc(4)])
}

test("catalog selection is exact and rejects malformed, ambiguous or unsafe declarations", async () => {
  const expected = await Effect.runPromise(declaredModelFiles(bundle({ entry }), "model:gguf:q4"))
  expect(expected.files).toHaveLength(2)
  for (const input of [Buffer.from("invalid"), bundle({}), bundle({ first: entry, second: entry }),
    bundle({ entry: { ...entry, target: { package: { source, files: [{ ...file, path: "../outside" }] } } } })]) {
    expect((await Effect.runPromise(declaredModelFiles(input, "model:gguf:q4").pipe(Effect.either)))._tag).toBe("Left")
  }
  const draft = await Effect.runPromise(declaredModelFiles(bundle({ entry: { ...entry, draft: { package: { source: { ...source, repository: "owner/draft" }, files: [file] } } } }), "model:gguf:q4"))
  expect(draft.files).toHaveLength(3)
})

for (const mode of ["valid", "corrupt", "missing-companion", "escaped-link", "linked-store"] as const) test(`model bytes match the admitted catalog: ${mode}`, () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-model-files-" }), profile = join(root, "profile")
  const repository = join(profile, "models", "hub", "models--owner--model"), snapshot = join(repository, "snapshots", source.revision)
  const blob = join(repository, "blobs", "weight"), outside = join(root, "outside")
  yield* fs.makeDirectory(join(snapshot, "nested"), { recursive: true })
  yield* fs.makeDirectory(join(repository, "blobs"), { recursive: true })
  yield* fs.writeFile(blob, mode === "corrupt" ? new TextEncoder().encode("wrong model bytes") : bytes)
  yield* fs.writeFile(outside, bytes)
  yield* fs.symlink(mode === "escaped-link" ? outside : blob, join(snapshot, "weights.gguf"))
  if (mode !== "missing-companion") yield* fs.writeFile(join(snapshot, "nested", "projector.gguf"), companion)
  if (mode === "linked-store") {
    yield* fs.rename(join(profile, "models"), join(root, "other-store"))
    yield* fs.symlink(join(root, "other-store"), join(profile, "models"))
  }
  const expected = yield* declaredModelFiles(bundle({ entry }), "model:gguf:q4")
  const result = yield* verifyModelFiles(profile, expected).pipe(Effect.either)
  expect(result._tag).toBe(mode === "valid" ? "Right" : "Left")
  if (mode === "missing-companion" && result._tag === "Left") expect(result.left._tag).toBe("AssertionFailure")
})).pipe(Effect.provide(BunContext.layer))))
