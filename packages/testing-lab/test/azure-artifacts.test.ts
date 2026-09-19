import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema, Stream } from "effect"
import { expect, test } from "vitest"
import { ArtifactStore } from "../src/artifact-store"
import { azureArtifactStore } from "../src/providers/azure-artifacts"
import { ProcessExecutor } from "../src/process"
import { sha256 } from "../src/snapshot"

for (const mode of ["valid", "corrupt-download", "oversized"] as const) test(`Azure artifacts verify bytes and clean staging: ${mode}`, () => Effect.runPromise(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const bytes = new TextEncoder().encode("private candidate bytes")
  const digest = sha256(bytes)
  const files: string[] = [], commands: string[] = []
  let uploaded = false
  const program = Effect.gen(function* () {
    const store = yield* ArtifactStore
    const invalid = yield* store.put(digest, Stream.make(new Uint8Array([0]))).pipe(Effect.either)
    expect(invalid._tag).toBe("Left")
    expect(commands).toEqual([])
    yield* store.put(digest, Stream.make(bytes))
    yield* store.put(digest, Stream.make(bytes))
    expect(commands.filter(command => command === "upload")).toHaveLength(1)
    expect(yield* store.exists(digest)).toBe(true)
    const downloaded = yield* store.get(digest).pipe(Stream.runCollect, Effect.either)
    expect(downloaded._tag).toBe(mode === "valid" ? "Right" : "Left")
    if (downloaded._tag === "Right") expect(Buffer.concat(Array.from(downloaded.right))).toEqual(Buffer.from(bytes))
    if (mode === "oversized") expect(commands).not.toContain("download")
    for (const file of files) expect(yield* fs.exists(file)).toBe(false)
  }).pipe(Effect.provide(azureArtifactStore({ executable: "az", subscription: "5304c4b3-d605-4193-b0cb-766c065acfa6", account: "fixtureaccount", container: "artifacts", maxBytes: 1024 })))
  yield* program.pipe(Effect.provideService(ProcessExecutor, { run: spec => Effect.gen(function* () {
    const op = spec.args[2]!
    commands.push(op)
    const value = (flag: string) => spec.args[spec.args.indexOf(flag) + 1]!
    expect(value("--auth-mode")).toBe("login")
    expect(value("--subscription")).toBe("5304c4b3-d605-4193-b0cb-766c065acfa6")
    expect(value("--name")).toBe(digest)
    if (op === "upload" || op === "download") {
      const file = value("--file"); files.push(file)
      if (op === "upload") { expect(sha256(yield* fs.readFile(file).pipe(Effect.orDie))).toBe(digest); expect(value("--overwrite")).toBe("false"); uploaded = true }
      else { expect(value("--if-match")).toBe("fixture-etag"); const received = bytes.slice(); if (mode === "corrupt-download") received[0] ^= 255; yield* fs.writeFile(file, received).pipe(Effect.orDie) }
    }
    const body = op === "exists" ? { exists: uploaded } : op === "show" ? { properties: { contentLength: mode === "oversized" ? 2048 : bytes.length, etag: "fixture-etag" } } : {}
    return { exitCode: 0, stdout: yield* Schema.encode(Schema.parseJson(Schema.Unknown))(body).pipe(Effect.orDie), stderr: "" }
  }) }))
}).pipe(Effect.provide(BunContext.layer))))
