import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option, Schema } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"
import { makePreparedUpdateStore, PreparedUpdate } from "./prepared-update"
import { PrivateFilePermissions } from "./private-files"

const key = generateKeyPairSync("ed25519")
const bytes = Buffer.from("a complete verified installer")
const target = { os: "linux" as const, arch: "arm64" as const, package: "deb" as const }
const release = await Effect.runPromise(signUpdateRelease({ version: "2.0.0", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }, target, key.privateKey))
let root: string, archive: string
const make = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  return yield* makePreparedUpdateStore({ dataDirectory: root, target, trustedPublishers: new Map([["publisher", key.publicKey]]) }).pipe(Effect.provideService(PrivateFilePermissions, {
    prepareDirectory: path => fs.makeDirectory(path, { recursive: true, mode: 0o700 }).pipe(Effect.orDie),
    createFile: path => fs.writeFileString(path, "", { flag: "wx", mode: 0o600 }).pipe(Effect.orDie),
    protectFile: () => Effect.void,
  }))
})
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem>) => Effect.runPromise(effect.pipe(Effect.provide(BunContext.layer)))
beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "prepared-update-")); archive = join(root, "download"); await writeFile(archive, bytes) })
afterEach(async () => rm(root, { recursive: true, force: true }))

describe("one durable prepared update", () => {
  it("persists verified bytes and only the release and installation union across reconstruction", async () => {
    await run(Effect.gen(function* () { const store = yield* make; yield* store.prepare(archive, release) }))
    expect((await readdir(join(root, "updates"))).sort()).toEqual(["magnitude.deb", "update.json"])
    const saved = JSON.parse(await readFile(join(root, "updates/update.json"), "utf8"))
    expect(saved).toEqual({ release, installation: { _tag: "Unattempted" } })
    await run(Effect.gen(function* () {
      const reopened = yield* make
      expect(Option.getOrThrow(yield* reopened.read)).toEqual(saved)
      expect(yield* reopened.verify(release)).toBe(join(root, "updates/magnitude.deb"))
      yield* reopened.recordAttempt(release)
    }))
    await run(Effect.gen(function* () {
      const reopened = yield* make
      expect(Option.getOrThrow(yield* reopened.read).installation).toEqual({ _tag: "Attempted" })
    }))
  })
  it("retains failed installer bytes for explicit retry without redownloading", async () => {
    await run(Effect.gen(function* () {
      const store = yield* make
      yield* store.prepare(archive, release)
      yield* store.recordAttempt(release)
      yield* store.recordFailure(release, "Authorization was cancelled")
      expect(Option.getOrThrow(yield* store.read).installation).toEqual({ _tag: "Failed", reason: "Authorization was cancelled" })
      yield* store.verify(release)
      yield* store.recordAttempt(release)
      expect(Option.getOrThrow(yield* store.read).installation).toEqual({ _tag: "Attempted" })
    }))
    expect(await readFile(join(root, "updates/magnitude.deb"))).toEqual(bytes)
  })
  it("rejects corruption before publication and when revalidating a saved installer", async () => {
    await writeFile(archive, Buffer.alloc(bytes.length))
    await run(Effect.gen(function* () {
      const store = yield* make
      expect((yield* store.prepare(archive, release).pipe(Effect.either))._tag).toBe("Left")
      expect(Option.isNone(yield* store.read)).toBe(true)
    }))
    expect(await readdir(join(root, "updates"))).toEqual([])
    await writeFile(archive, bytes)
    await run(Effect.gen(function* () { const store = yield* make; yield* store.prepare(archive, release) }))
    await writeFile(join(root, "updates/magnitude.deb"), Buffer.alloc(bytes.length))
    await run(Effect.gen(function* () {
      const store = yield* make
      expect((yield* store.verify(release).pipe(Effect.either))._tag).toBe("Left")
      expect(Option.getOrThrow(yield* store.read).installation._tag).toBe("Unattempted")
    }))
  })
  it("rejects an old helper's write after the pending release changes", async () => {
    const next = { ...release, signature: "A".repeat(86) + "==" }
    await run(Effect.gen(function* () {
      const store = yield* make
      yield* store.prepare(archive, release)
      expect((yield* store.recordFailure(next, "stale helper").pipe(Effect.either))._tag).toBe("Left")
      expect(Option.getOrThrow(yield* store.read).installation._tag).toBe("Unattempted")
      expect((yield* store.prepare(archive, release).pipe(Effect.either))._tag).toBe("Left")
    }))
  })
  it("cleans an abandoned transfer and orphan installer without touching identity or config", async () => {
    await writeFile(join(root, "identity.pem"), "private identity")
    await writeFile(join(root, "config.json"), "{}")
    await run(Effect.gen(function* () { const store = yield* make; yield* store.prepare(archive, release); yield* store.discard }))
    await writeFile(join(root, "updates/installer-12345678-1234-1234-1234-123456789abc.tmp"), "partial")
    await writeFile(join(root, "updates/magnitude.deb"), "orphan")
    await run(Effect.gen(function* () { const store = yield* make; yield* store.removeAbandonedTransfers }))
    expect(await readdir(join(root, "updates"))).toEqual([])
    expect(await readFile(join(root, "identity.pem"), "utf8")).toBe("private identity")
    expect(await readFile(join(root, "config.json"), "utf8")).toBe("{}")
  })
  it("rejects duplicate state fields, unsupported modes and unbounded failure messages", () => {
    for (const installation of [{ _tag: "Downloading" }, { _tag: "Failed", reason: "x".repeat(501) }, { _tag: "Attempted", version: "2.0.0" }]) {
      expect(Schema.decodeUnknownEither(PreparedUpdate)({ release, installation }, { onExcessProperty: "error" })._tag).toBe("Left")
    }
  })
})
