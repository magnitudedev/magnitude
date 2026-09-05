import { BunContext } from "@effect/platform-bun"
import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Either, Option, Schema } from "effect"
import { parse } from "jsonc-parser"
import { describe, expect, it } from "vitest"
import { decodePiSettings, piPackageFilters, piPackageSource, readPiSettings, replacePiPackages } from "./pi-settings"

const path = "/pi/settings.json"

describe("Pi settings boundary", () => {
  it("decodes both package representations and preserves absent versus empty filters", async () => {
    const { settings } = await Effect.runPromise(decodePiSettings(`{
      // User settings may contain comments and trailing commas.
      "packages": ["npm:example", {"source":"./local"}, {"source":"./disabled","autoload":false,"extensions":[]}],
      "defaultProvider": "user",
      "defaultModel": "user/model",
    }`, path))
    const packages = Option.getOrThrow(settings.packages)
    expect(packages.map(piPackageSource)).toEqual(["npm:example", "./local", "./disabled"])
    expect(packages.map(piPackageFilters)).toEqual([Option.none(), Option.none(), Option.some([])])
    expect(settings.defaultProvider).toEqual(Option.some("user"))
    expect(settings.defaultModel).toEqual(Option.some("user/model"))
  })

  it.each([
    ["missing packages", "{}", Option.none()],
    ["empty packages", '{"packages":[]}', Option.some([])],
  ])("preserves %s", async (_name, text, expected) => {
    const document = await Effect.runPromise(decodePiSettings(text, path))
    expect(document.settings.packages).toEqual(expected)
    const restored = await Effect.runPromise(replacePiPackages(document, document.settings.packages))
    expect(parse(restored)).toEqual(parse(text))
  })

  it.each([
    ["malformed JSONC", '{"packages": [}', "offset"],
    ["non-object root", "[]", "Expected"],
    ["null root", "null", "Expected"],
    ["non-array packages", '{"packages":{}}', "packages"],
    ["null packages", '{"packages":null}', "packages"],
    ["invalid entry", '{"packages":[42]}', "packages"],
    ["missing source", '{"packages":[{"extensions":[]}]}', "source"],
    ["empty source", '{"packages":[""]}', "packages"],
    ["invalid autoload", '{"packages":[{"source":"local","autoload":"false"}]}', "autoload"],
    ["non-array filters", '{"packages":[{"source":"local","extensions":"*.js"}]}', "extensions"],
    ["invalid filter", '{"packages":[{"source":"local","extensions":[42]}]}', "extensions"],
    ["null filters", '{"packages":[{"source":"local","extensions":null}]}', "extensions"],
    ["invalid selection", '{"defaultModel":42}', "defaultModel"],
  ])("rejects %s as a typed error with the file and field", async (_name, text, detail) => {
    const result = await Effect.runPromise(Effect.either(decodePiSettings(text, path)))
    expect(Either.isLeft(result)).toBe(true)
    if (Either.isLeft(result)) {
      expect(result.left._tag).toBe("PiSettingsInvalid")
      expect(result.left.message).toContain(path)
      expect(result.left.detail).toContain(detail)
    }
  })

  it("preserves unknown package fields when recovery rewrites the package array", async () => {
    const original = {
      source: "./local",
      autoload: false,
      extensions: [],
      prompts: ["./prompts"],
      future: { nested: [1, null, { preserve: true }] },
    }
    const text = Schema.encodeSync(Schema.parseJson())({ packages: [original, "npm:other"], theme: "dark" })
    const document = await Effect.runPromise(decodePiSettings(text, path))
    const restored = await Effect.runPromise(replacePiPackages(document, document.settings.packages))
    expect(parse(restored)).toEqual({ packages: [original, "npm:other"], theme: "dark" })
    expect(restored).not.toContain('"_tag"')
  })

  it("retains unrelated fields in the decoded snapshot for recovery comparisons", async () => {
    const before = await Effect.runPromise(decodePiSettings('{"theme":"before"}', path))
    const after = await Effect.runPromise(decodePiSettings('{"theme":"after"}', path))
    expect(before.settings).not.toEqual(after.settings)
  })

  it("treats only a missing file as empty settings", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-settings-" })
      const file = `${root}/settings.json`
      const absent = yield* readPiSettings(file)
      expect(absent.settings.packages).toEqual(Option.none())
      yield* fs.writeFileString(file, "")
      const empty = yield* Effect.either(readPiSettings(file))
      expect(Either.isLeft(empty)).toBe(true)
    }).pipe(Effect.provide(BunContext.layer))))
  })
})
