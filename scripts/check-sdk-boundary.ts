import { Console, Effect } from "effect"
import { resolve } from "node:path"

/** Inspect the entire exported SDK, not only the subset a particular plugin uses. */
const program = Effect.gen(function* () {
  const result = yield* Effect.tryPromise(() => Bun.build({
    entrypoints: [resolve(import.meta.dir, "../packages/sdk/src/index.ts")],
    target: "browser", format: "esm", metafile: true,
    external: ["effect", "@effect/platform", "@effect/platform/*", "@effect/rpc", "@effect/rpc/*"],
  }))
  if (!result.success) return yield* Effect.dieMessage(result.logs.map(String).join("\n"))
  if (!result.metafile) return yield* Effect.dieMessage("SDK dependency audit requires build metadata")
  const paths = Object.keys(result.metafile.inputs)
  const forbidden = paths.filter(path => /(?:packages\/(?:daemon-management|providers|storage|effect-query|icn|agent)\/|@effect\/(?:platform-bun|platform-node)|@effect-atom\/|acn-protocol\/src\/coordination\/|bun:|node:)/.test(path))
  const external = Object.values(result.metafile.outputs).flatMap(output => output.imports.map(input => input.path))
  if (forbidden.length || external.some(path => /^(?:node:|bun:|@magnitudedev\/)/.test(path))) return yield* Effect.dieMessage(`Impure SDK runtime imports:\n${[...forbidden,...external.filter(path => /^(?:node:|bun:|@magnitudedev\/)/.test(path))].join("\n")}`)
  yield* Console.log(`SDK browser bundle verified: ${paths.length} reachable modules; no daemon management, SQLite, query cache, provider implementations, or native platform adapters.`)
})
await Effect.runPromise(program)
