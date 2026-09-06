import { BunContext } from "@effect/platform-bun"
import * as FileSystem from "@effect/platform/FileSystem"
import { Context, Effect, JSONSchema, Schema } from "effect"
import { inspectHermesPluginContent } from "../../../packages/release/src/hermes-plugin-content"
import { Models, Inference, AcnRpcRecoveryPolicyTag, MAGNITUDE_RPC_VERSION, AcnHealthResponseSchema, MagnitudeHealthResponseSchema } from "../../../packages/acn-protocol/src"

// Hermes runs Python. Export encoded boundary schemas, never a parallel handwritten
// model contract. Domain refinements remain authoritative on the ACN server.
const wire = (schema: Schema.Schema.Any) => JSONSchema.make(Schema.encodedSchema(schema))

await Effect.runPromise(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = new URL("..", import.meta.url).pathname.replace(/\/$/, "")
  const operations = Object.fromEntries(Object.entries({
    status: Models.getCatalog,
    load: Models.load,
    stop: Models.stop,
    observations: Inference.getObservations,
  }).map(([name, rpc]) => [name, {
    tag: rpc._tag,
    recovery: Context.get(rpc.annotations, AcnRpcRecoveryPolicyTag),
    payload: wire(rpc.payloadSchema),
    success: wire(rpc.successSchema),
    error: wire(rpc.errorSchema),
  }]))
  yield* fs.makeDirectory(`${root}/dist/skills/magnitude`, { recursive: true })
  yield* fs.writeFileString(`${root}/dist/rpc-contract.json`, JSON.stringify({
    rpcVersion: MAGNITUDE_RPC_VERSION,
    health: wire(Schema.Union(AcnHealthResponseSchema, MagnitudeHealthResponseSchema)),
    operations,
  }, null, 2) + "\n")
  yield* fs.copyFile(new URL("../../../cli/src/harness-connections/magnitude-skill.md", import.meta.url).pathname,
    `${root}/dist/skills/magnitude/SKILL.md`)
  const desktop = yield* Effect.tryPromise(() => Bun.build({
    entrypoints: [`${root}/desktop-src/plugin.tsx`],
    outdir: `${root}/desktop`,
    target: "browser",
    format: "esm",
    jsx: { runtime: "automatic", importSource: "react", development: false },
    env: "disable",
    define: { "process.env.NODE_ENV": '"production"' },
    external: ["@hermes/plugin-sdk", "react", "react/jsx-runtime"],
  }))
  if (!desktop.success) return yield* Effect.fail(new Error(desktop.logs.map(String).join("\n")))
  const { metadata } = yield* inspectHermesPluginContent(root, MAGNITUDE_RPC_VERSION)
  yield* fs.writeFileString(`${root}/dist/magnitude-plugin.json`, JSON.stringify(metadata, null, 2) + "\n")
}).pipe(Effect.provide(BunContext.layer)))
