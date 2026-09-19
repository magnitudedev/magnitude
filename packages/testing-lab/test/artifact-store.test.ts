import { expect, test } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Stream } from "effect"
import { join } from "node:path"
import { ArtifactStore, downloadObject, fileArtifactStore } from "../src/artifact-store"
import { sha256 } from "../src/snapshot"

test("streamed objects reject corrupt uploads and downloads without publishing partial files", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-object-test-" })
  yield* Effect.gen(function* () {
    const store = yield* ArtifactStore
    const data = new TextEncoder().encode("expected content")
    const digest = sha256(data)
    yield* store.put(digest, Stream.fromIterable([data.slice(0, 4), data.slice(4)]))
    expect(yield* store.exists(digest)).toBe(true)
    const path = join(root, "consumer")
    yield* downloadObject(digest, path)
    expect(yield* fs.readFileString(path)).toBe("expected content")
    expect(yield* store.put(digest, Stream.make(new TextEncoder().encode("corrupt"))).pipe(Effect.either)).toMatchObject({ _tag: "Left" })
    expect(yield* fs.readFileString(join(root, "objects", digest))).toBe("expected content")
    yield* fs.writeFileString(join(root, "objects", digest), "corrupt store")
    expect(yield* downloadObject(digest, path).pipe(Effect.either)).toMatchObject({ _tag: "Left" })
    expect(yield* fs.readFileString(path)).toBe("expected content")
    expect((yield* fs.readDirectory(root)).some(p => p.includes(".download-"))).toBe(false)
    expect((yield* fs.readDirectory(join(root, "objects"))).some(p => p.startsWith(".upload-"))).toBe(false)
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(BunContext.layer))))
