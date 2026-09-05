import { Effect, Schema } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import { inspectPluginContent, PLUGIN_METADATA_PATH } from "@magnitudedev/release/plugin-content"
import { PluginContentManifestSchema } from "@magnitudedev/release/plugins"

// Effect and host/platform packages remain external. SDK and all private reachable workspace code are bundled.
await Effect.runPromise(Effect.tryPromise(() => Bun.build({
  entrypoints: [new URL("../extensions/magnitude.ts", import.meta.url).pathname],
  outdir: new URL("../dist", import.meta.url).pathname,
  target: "node",
  format: "esm",
  external: ["effect", "@effect/platform", "@effect/platform/*", "@effect/platform-node", "@effect/platform-node/*", "@effect/rpc", "@effect/rpc/*", "@earendil-works/pi-ai", "@earendil-works/pi-ai/*", "@earendil-works/pi-coding-agent", "@earendil-works/pi-tui"],
})).pipe(Effect.flatMap((result) => result.success
  ? Effect.void
  : Effect.fail(new Error(result.logs.map(String).join("\n"))))))

await Effect.runPromise(Effect.gen(function* () {
  const directory = new URL("..", import.meta.url).pathname.replace(/\/$/, "")
  const { metadata } = yield* inspectPluginContent(directory, MAGNITUDE_RPC_VERSION)
  const fs = yield* FileSystem.FileSystem
  yield* fs.writeFileString(`${directory}/${PLUGIN_METADATA_PATH}`, `${yield* Schema.encode(Schema.parseJson(PluginContentManifestSchema, { space: 2 }))(metadata)}\n`)
}).pipe(Effect.provide(BunContext.layer)))
