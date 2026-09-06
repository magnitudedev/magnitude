import * as FileSystem from "@effect/platform/FileSystem"
import * as Command from "@effect/platform/Command"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { dirname } from "node:path"
import { expect, it } from "vitest"
import { HERMES_PLUGIN_FILES, inspectHermesPluginContent } from "./hermes-plugin-content"
import { packHermesPlugin, verifyHermesPluginArtifact } from "./hermes-plugin-artifact"

it("produces the same immutable native Git package and verifies its content and exact revision", async () => {
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-git-test-" })
    const source = `${root}/source`
    for (const file of HERMES_PLUGIN_FILES) {
      yield* fs.makeDirectory(dirname(`${source}/${file}`), { recursive: true })
      yield* fs.writeFileString(`${source}/${file}`, file === "plugin.yaml"
        ? "name: magnitude\nversion: 0.0.1\n" : `fixture ${file}\n`)
    }
    const { metadata } = yield* inspectHermesPluginContent(source, 1)
    yield* fs.writeFileString(`${source}/dist/magnitude-plugin.json`, JSON.stringify(metadata))
    yield* fs.writeFileString(`${source}/development-only.txt`, "not included")
    const first = yield* packHermesPlugin(source, `${root}/first`)
    const second = yield* packHermesPlugin(source, `${root}/second`)
    expect(second).toEqual(first)
    expect(yield* verifyHermesPluginArtifact(first, `${root}/first`)).toBe(`${root}/first/${first.filename}`)
    expect((yield* Effect.either(verifyHermesPluginArtifact({ ...first, revision: "0".repeat(40) }, `${root}/first`)))._tag).toBe("Left")
    expect((yield* Effect.either(verifyHermesPluginArtifact({ ...first, rpcVersion: 2 }, `${root}/first`)))._tag).toBe("Left")
    // Git transport encoding is not release identity. A different pack of the
    // exact same commit must still verify against the allocated artifact.
    const runGit = (...args: string[]) => Command.make("git", ...args).pipe(Command.exitCode)
    expect(Number(yield* runGit("clone", `${root}/first/${first.filename}`, `${root}/repacked-source`))).toBe(0)
    yield* fs.makeDirectory(`${root}/repacked`)
    expect(Number(yield* runGit("-C", `${root}/repacked-source`, "-c", "pack.compression=0", "bundle", "create", `${root}/repacked/${first.filename}`, "refs/heads/main"))).toBe(0)
    expect(yield* verifyHermesPluginArtifact(first, `${root}/repacked`)).toBe(`${root}/repacked/${first.filename}`)
    yield* fs.writeFileString(`${root}/first/${first.filename}`, "tampered")
    expect((yield* Effect.either(verifyHermesPluginArtifact(first, `${root}/first`)))._tag).toBe("Left")
  })).pipe(Effect.provide(BunContext.layer)))
})
