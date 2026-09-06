import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { Effect, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { HERMES_PLUGIN_FILES, inspectHermesPluginContent, verifyHermesPluginContent, stageHermesPluginContent } from "./hermes-plugin-content"
import { PLUGIN_METADATA_PATH } from "./plugin-content"
import { PluginContentManifestSchema } from "./plugins"

describe("Hermes consumer content", () => {
  it("covers both native surfaces and detects modifications without coupling content identity to release version", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-content-" })
      for (const path of HERMES_PLUGIN_FILES) {
        const parent = path.slice(0, path.lastIndexOf("/"))
        if (path.includes("/")) yield* fs.makeDirectory(`${directory}/${parent}`, { recursive: true })
        yield* fs.writeFileString(`${directory}/${path}`, path === "plugin.yaml"
          ? "name: magnitude\nversion: 0.0.1\n" : `fixture ${path}`)
      }
      const initial = yield* inspectHermesPluginContent(directory, 1)
      const recorded = yield* Schema.encode(Schema.parseJson(PluginContentManifestSchema))(initial.metadata)
      yield* fs.writeFileString(`${directory}/${PLUGIN_METADATA_PATH}`, recorded)
      expect((yield* verifyHermesPluginContent(directory)).metadata).toEqual(initial.metadata)
      yield* fs.writeFileString(`${directory}/developer-notes.txt`, "not consumer content")
      const destination = `${directory}/consumer`
      expect((yield* stageHermesPluginContent(directory, destination)).metadata).toEqual(initial.metadata)
      expect(yield* fs.exists(`${destination}/developer-notes.txt`)).toBe(false)
      expect((yield* Effect.either(stageHermesPluginContent(directory, destination)))._tag).toBe("Left")
      yield* fs.writeFileString(`${directory}/plugin.yaml`, "name: magnitude\nversion: 0.0.2\n")
      const versioned = yield* inspectHermesPluginContent(directory, 1)
      expect(versioned.metadata.contentFingerprint).toBe(initial.metadata.contentFingerprint)
      expect((yield* Effect.either(verifyHermesPluginContent(directory)))._tag).toBe("Left")
      expect((yield* inspectHermesPluginContent(directory, 2)).metadata.contentFingerprint)
        .not.toBe(initial.metadata.contentFingerprint)
      for (const path of ["progress.py", "desktop/plugin.js", "dashboard/plugin_api.py", "dist/skills/magnitude/SKILL.md"]) {
        yield* fs.writeFileString(`${directory}/${path}`, "changed")
        expect((yield* inspectHermesPluginContent(directory, 1)).metadata.contentFingerprint)
          .not.toBe(initial.metadata.contentFingerprint)
        yield* fs.writeFileString(`${directory}/${path}`, `fixture ${path}`)
      }
      yield* fs.writeFileString(`${directory}/plugin.yaml`, "name: magnitude\nversion: 0.0.2\nrequires:\n  hermes: '>=0.21.0'\n")
      expect((yield* inspectHermesPluginContent(directory, 1)).metadata.contentFingerprint)
        .not.toBe(initial.metadata.contentFingerprint)
    })).pipe(Effect.provide(BunContext.layer)))
  })
})
