import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { ArtifactStore, fileArtifactStore } from "../src/artifact-store"
import { publishEvidenceFile } from "../src/evidence"
import { sha256 } from "../src/snapshot"

test("retains binary traces by content address after the worker file is deleted", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-evidence-" })
  const evidence = join(root, "evidence")
  yield* fs.makeDirectory(join(evidence, "desktop"), { recursive: true })
  const bytes = new Uint8Array([80, 75, 3, 4, 0, 255, 128, 1])
  yield* fs.writeFile(join(evidence, "desktop", "ui-trace.zip"), bytes)
  yield* Effect.gen(function* () {
    const result = yield* publishEvidenceFile(evidence, "desktop/ui-trace.zip", 1024)
    expect(result).toEqual({ path: "evidence/desktop/ui-trace.zip", bytes: bytes.length, sha256: sha256(bytes) })
    yield* fs.remove(evidence, { recursive: true })
    const store = yield* ArtifactStore
    const recovered = yield* store.get(result.sha256).pipe(Stream.runCollect)
    expect(Buffer.concat(Array.from(recovered))).toEqual(Buffer.from(bytes))
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(BunContext.layer))))

test("rejects traversal, symlinks, directories and oversized evidence", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-evidence-invalid-" })
  const evidence = join(root, "evidence")
  yield* fs.makeDirectory(evidence)
  yield* fs.writeFileString(join(root, "outside"), "private")
  yield* fs.writeFileString(join(evidence, "large"), "too large")
  yield* fs.symlink(join(root, "outside"), join(evidence, "linked"))
  yield* fs.makeDirectory(join(evidence, "directory"))
  yield* Effect.gen(function* () {
    for (const path of ["../outside", join(root, "outside"), "linked", "directory", "large"]) {
      expect((yield* publishEvidenceFile(evidence, path, 4).pipe(Effect.either))._tag).toBe("Left")
    }
    expect(yield* fs.readDirectory(join(root, "objects"))).toEqual([])
  }).pipe(Effect.provide(fileArtifactStore(join(root, "objects"))))
})).pipe(Effect.provide(BunContext.layer))))
