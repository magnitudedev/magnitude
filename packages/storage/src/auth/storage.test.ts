import { BunFileSystem, BunPath } from "@effect/platform-bun"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import { Effect, Layer, Option, Redacted } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtemp, rm, stat, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services"
import type { HuggingFaceCredential } from "../types"
import { makeAuthStorage } from "./storage"

const credential: HuggingFaceCredential = {
  type: "oauth",
  accessToken: Redacted.make("hf_oauth_access"),
  refreshToken: Option.some(Redacted.make("hf_oauth_refresh")),
  expiresAt: Option.some(1_900_000_000_000),
  username: "ada",
  avatarUrl: Option.none(),
}

describe("auth storage", () => {
  let root: string

  const run = <A, E>(effect: Effect.Effect<A, E, GlobalStorage | FileSystem.FileSystem | Path.Path>) =>
    Effect.runPromise(effect.pipe(Effect.provide(Layer.mergeAll(
      BunFileSystem.layer,
      BunPath.layer,
      Layer.succeed(GlobalStorage, GlobalStorage.of({ root, paths: makeGlobalStoragePaths(root) })),
    ))))

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "magnitude-auth-"))
  })

  afterEach(async () => {
    await rm(root, { recursive: true, force: true })
  })

  it("keeps the Hugging Face account in its own owner-only file, out of auth.json", async () => {
    const { providers, huggingFace } = await run(Effect.gen(function* () {
      const auth = yield* makeAuthStorage()
      yield* auth.set("anthropic", { type: "api", key: "sk-test" })
      yield* auth.setHuggingFace(credential)
      return { providers: yield* auth.loadAll(), huggingFace: yield* auth.getHuggingFace }
    }))
    expect(Object.keys(providers)).toEqual(["anthropic"])
    expect(Option.map(huggingFace, (stored) => Redacted.value(stored.accessToken))).toEqual(Option.some("hf_oauth_access"))
    const paths = makeGlobalStoragePaths(root)
    expect((await stat(paths.huggingFaceFile)).mode & 0o777).toBe(0o600)
    expect(await Bun.file(paths.huggingFaceFile).json()).toMatchObject({ type: "oauth", accessToken: "hf_oauth_access", username: "ada" })
    expect(await Bun.file(paths.authFile).text()).not.toContain("hf_oauth")
  })

  it("deletes the file on sign-out and leaves providers alone", async () => {
    const paths = makeGlobalStoragePaths(root)
    const afterSignOut = await run(Effect.gen(function* () {
      const auth = yield* makeAuthStorage()
      yield* auth.set("anthropic", { type: "api", key: "sk-test" })
      yield* auth.setHuggingFace(credential)
      yield* auth.removeHuggingFace
      return { providers: yield* auth.loadAll(), huggingFace: yield* auth.getHuggingFace }
    }))
    expect(Object.keys(afterSignOut.providers)).toEqual(["anthropic"])
    expect(Option.isNone(afterSignOut.huggingFace)).toBe(true)
    expect(await Bun.file(paths.huggingFaceFile).exists()).toBe(false)
  })

  it("reads a missing file as signed out and reports an invalid one", async () => {
    const read = () => run(Effect.flatMap(makeAuthStorage(), (auth) => auth.getHuggingFace.pipe(Effect.either)))
    const missing = await read()
    expect(missing._tag === "Right" && Option.isNone(missing.right)).toBe(true)
    await writeFile(makeGlobalStoragePaths(root).huggingFaceFile, JSON.stringify({ type: "oauth" }))
    expect((await read())._tag).toBe("Left")
  })

  it("never prints a token when the credential is inspected", () => {
    expect(JSON.stringify(credential)).not.toContain("hf_oauth")
    expect(String(credential.accessToken)).not.toContain("hf_oauth")
  })
})
