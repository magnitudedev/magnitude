import * as FileSystem from "@effect/platform/FileSystem"
import { Effect } from "effect"
import { dirname } from "node:path"
import { HERMES_PLUGIN_FILES, inspectHermesPluginContent } from "./hermes-plugin-content"

/** Minimal consumer files for release contract tests; never shipped. */
export const writeHermesPluginFixture = (source: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  for (const file of HERMES_PLUGIN_FILES) {
    yield* fs.makeDirectory(dirname(`${source}/${file}`), { recursive: true })
    yield* fs.writeFileString(`${source}/${file}`, file === "plugin.yaml"
      ? "name: magnitude\nversion: 0.0.1\n" : `fixture ${file}\n`)
  }
  const { metadata } = yield* inspectHermesPluginContent(source, 1)
  yield* fs.writeFileString(`${source}/dist/magnitude-plugin.json`, JSON.stringify(metadata))
})
