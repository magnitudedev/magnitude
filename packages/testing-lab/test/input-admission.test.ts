import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Layer, Stream } from "effect"
import { join } from "node:path"
import { expect, test } from "vitest"
import { fileArtifactStore } from "../src/artifact-store"
import { Database, initializeDatabase } from "../src/database"
import { OwnerId } from "../src/domain"
import { InputRegistry, InputRegistryLive } from "../src/inputs"
import { ProcessExecutorLive } from "../src/process"
import { manifestJson, RelativePath, sha256, type SourceManifest } from "../src/snapshot"
import { temporaryDatabase } from "./postgres"

test("large source graphs batch admission without accepting missing, foreign or conflicting object lengths", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "lab-input-admission-" })
  const database = yield* temporaryDatabase
  yield* Effect.gen(function* () {
    yield* initializeDatabase
    const db = yield* Database
    for (const mode of ["valid", "missing", "foreign", "length", "conflicting-reference"] as const) {
      const owner = OwnerId.make(mode)
      const entries: SourceManifest["entries"] = Array.from({ length: 1001 }, (_, index) => ({ kind: "file", path: RelativePath.make(`file-${index}`),
        sha256: sha256(`content-${index}`), bytes: Buffer.byteLength(`content-${index}`), executable: false }))
      // The object registry is authoritative after upload verification; seed its metadata as a DB fixture.
      const last = entries[1000]!
      const admitted = entries.filter((_, index) => mode !== "missing" || index !== 1000)
      yield* db.query(`INSERT INTO lab_objects(owner,digest,bytes) SELECT * FROM unnest($1::text[],$2::text[],$3::bigint[])`, [
        admitted.map((_, index) => mode === "foreign" && index === 1000 ? "another-owner" : owner),
        admitted.map(entry => entry.kind === "file" ? entry.sha256 : ""),
        admitted.map((entry, index) => entry.kind === "file" ? entry.bytes + (mode === "length" && index === 1000 ? 1 : 0) : 0),
      ])
      const manifest: SourceManifest = { schemaVersion: 1, kind: "source", commit: "a".repeat(40), entries: mode === "conflicting-reference" && last.kind === "file"
        ? [...entries, { ...last, path: RelativePath.make("conflicting-copy"), bytes: last.bytes + 1 }] : entries }
      const json = manifestJson(manifest), digest = sha256(json)
      let batches = 0
      const observed = Layer.succeed(Database, { ...db, query: (sql, values) => {
        if (sql.startsWith("SELECT digest,bytes")) batches++
        return db.query(sql, values)
      } })
      yield* Effect.gen(function* () {
        const inputs = yield* InputRegistry
        yield* inputs.upload(owner, digest, Stream.make(new TextEncoder().encode(json)))
        const result = yield* inputs.register(owner, { kind: "source", digest }).pipe(Effect.either)
        expect(result._tag, mode).toBe(mode === "valid" ? "Right" : "Left")
        if (result._tag === "Left") expect(result.left._tag).toBe(mode === "missing" || mode === "foreign" ? "InputDenied" : "InvalidInput")
        expect((yield* inputs.require(owner, { kind: "source", digest }).pipe(Effect.either))._tag).toBe(mode === "valid" ? "Right" : "Left")
        expect(batches).toBe(2)
      }).pipe(Effect.provide(InputRegistryLive.pipe(Layer.provide(Layer.merge(observed, fileArtifactStore(join(root, "objects")))))))
    }
  }).pipe(Effect.provide(database))
})).pipe(Effect.provide([BunContext.layer, ProcessExecutorLive]))))
