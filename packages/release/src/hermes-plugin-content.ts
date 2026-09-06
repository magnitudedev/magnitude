import * as FileSystem from "@effect/platform/FileSystem"
import { Effect, Schema } from "effect"
import { parse } from "yaml"
import { canonical } from "@magnitudedev/utils/canonical-key"
import { JsonValueSchema } from "@magnitudedev/utils/schema"
import { PluginContentManifestSchema } from "./plugins"
import { packageContentFingerprint, PLUGIN_METADATA_PATH, PluginContentMismatch, sha256 } from "./plugin-content"

export const HERMES_PLUGIN_IDENTITY = "@magnitudedev/hermes-companion"
export const HERMES_PLUGIN_FILES = [
  "plugin.yaml", "__init__.py", "client.py", "commands.py", "observations.py", "progress.py", "terminal.py",
  "after-install.md", "README.md", "dashboard/manifest.json", "dashboard/plugin_api.py", "desktop/plugin.js",
  "dist/rpc-contract.json", "dist/skills/magnitude/SKILL.md",
] as const
const Manifest = Schema.Struct({ name: Schema.Literal("magnitude"), version: Schema.NonEmptyString })
const JsonObject = Schema.Record({ key: Schema.String, value: JsonValueSchema })

export const inspectHermesPluginContent = (directory: string, rpcVersion: number) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const source = yield* fs.readFileString(`${directory}/plugin.yaml`)
  const raw = yield* Effect.try(() => parse(source)).pipe(Effect.flatMap(Schema.decodeUnknown(JsonObject)))
  const manifest = yield* Schema.decodeUnknown(Manifest)(raw)
  const files = Object.fromEntries(yield* Effect.forEach(HERMES_PLUGIN_FILES, path =>
    fs.readFile(`${directory}/${path}`).pipe(Effect.map(bytes => [path, sha256(bytes)] as const))))
  // Native installation fields are fingerprinted as data, independently of the
  // assigned version; the actual YAML bytes remain covered by the file manifest.
  const contentFiles = Object.fromEntries(Object.entries(files).filter(([path]) => path !== "plugin.yaml"))
  const metadata = yield* Schema.decodeUnknown(PluginContentManifestSchema)({
    name: HERMES_PLUGIN_IDENTITY, version: manifest.version, rpcVersion, files,
    contentFingerprint: packageContentFingerprint(raw, rpcVersion, contentFiles),
  })
  return { manifest, metadata }
})

export const verifyHermesPluginContent = (directory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const recorded = yield* fs.readFileString(`${directory}/${PLUGIN_METADATA_PATH}`).pipe(
    Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(PluginContentManifestSchema))),
  )
  const actual = yield* inspectHermesPluginContent(directory, recorded.rpcVersion)
  if (canonical(recorded) !== canonical(actual.metadata)) return yield* new PluginContentMismatch({ directory })
  return actual
})

/** Stage only the verified consumer files into a caller-owned empty directory. */
export const stageHermesPluginContent = (source: string, destination: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const inspected = yield* verifyHermesPluginContent(source)
  yield* fs.makeDirectory(destination)
  for (const path of [...HERMES_PLUGIN_FILES, PLUGIN_METADATA_PATH]) {
    const parent = path.slice(0, path.lastIndexOf("/"))
    if (path.includes("/")) yield* fs.makeDirectory(`${destination}/${parent}`, { recursive: true })
    yield* fs.copyFile(`${source}/${path}`, `${destination}/${path}`)
  }
  // Detect source changes during staging before this directory can be accepted.
  const staged = yield* verifyHermesPluginContent(destination)
  if (canonical(staged.metadata) !== canonical(inspected.metadata)) {
    return yield* new PluginContentMismatch({ directory: destination })
  }
  return staged
})
