import { BunFileSystem, BunPath } from "@effect/platform-bun"
import type * as FileSystem from "@effect/platform/FileSystem"
import type * as Path from "@effect/platform/Path"
import { Effect, Layer, Option, Redacted } from "effect"
import { afterEach, beforeEach, describe, expect, it } from "vitest"
import { mkdtemp, rm, stat } from "node:fs/promises"
import { join } from "node:path"
import { tmpdir } from "node:os"
import { makeGlobalStoragePaths } from "../paths"
import { GlobalStorage } from "../services"
import type { HuggingFaceCredential } from "../types"
import { HUGGING_FACE_AUTH_KEY, makeAuthStorage } from "./storage"

const credential: HuggingFaceCredential = {
  type: "huggingface-oauth",
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

  it("keeps the Hugging Face account beside providers, owner-only, and out of the provider list", async () => {
    const { providers, huggingFace } = await run(Effect.gen(function* () {
      const auth = yield* makeAuthStorage()
      yield* auth.set("anthropic", { type: "api", key: "sk-test" })
      yield* auth.setHuggingFace(credential)
      yield* auth.set("openai", { type: "api", key: "sk-other" })
      return { providers: yield* auth.loadAll(), huggingFace: yield* auth.getHuggingFace }
    }))
    expect(Object.keys(providers).sort()).toEqual(["anthropic", "openai"])
    expect(Option.map(huggingFace, (stored) => Redacted.value(stored.accessToken))).toEqual(Option.some("hf_oauth_access"))
    const file = makeGlobalStoragePaths(root).authFile
    expect((await stat(file)).mode & 0o777).toBe(0o600)
    const raw = await Bun.file(file).json() as Record<string, { readonly accessToken?: string }>
    expect(raw[HUGGING_FACE_AUTH_KEY]?.accessToken).toBe("hf_oauth_access")
  })

  it("removes only the account, and the file once nothing is left", async () => {
    const file = makeGlobalStoragePaths(root).authFile
    const afterSignOut = await run(Effect.gen(function* () {
      const auth = yield* makeAuthStorage()
      yield* auth.set("anthropic", { type: "api", key: "sk-test" })
      yield* auth.setHuggingFace(credential)
      yield* auth.removeHuggingFace
      return { providers: yield* auth.loadAll(), huggingFace: yield* auth.getHuggingFace }
    }))
    expect(Object.keys(afterSignOut.providers)).toEqual(["anthropic"])
    expect(Option.isNone(afterSignOut.huggingFace)).toBe(true)
    await run(Effect.gen(function* () {
      const auth = yield* makeAuthStorage()
      yield* auth.remove("anthropic")
      yield* auth.setHuggingFace(credential)
      yield* auth.removeHuggingFace
    }))
    expect(await Bun.file(file).exists()).toBe(false)
  })

  it("never prints a token when the credential is inspected", () => {
    expect(JSON.stringify(credential)).not.toContain("hf_oauth")
    expect(String(credential.accessToken)).not.toContain("hf_oauth")
  })
})
