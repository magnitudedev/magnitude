import { Effect, Schema } from "effect"
import { FileSystem } from "@effect/platform"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { resolve, join } from "node:path"
import { createHash } from "node:crypto"

// Private deployment artifact: the landing repo consumes these exact shared protocol/server bytes.
class DistributionBuildFailed extends Schema.TaggedError<DistributionBuildFailed>()("DistributionBuildFailed", {}) {}
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const output = resolve(process.argv[2] ?? "dist/distribution-server")
  yield* fs.makeDirectory(output, { recursive: true })
  yield* Effect.tryPromise({ try: async () => {
    const build = await Bun.build({ entrypoints: [resolve(import.meta.dir, "../src/hosted-update/server.ts")], target: "node", format: "esm", outdir: output, external: ["pg"], minify: true })
    if (!build.success) throw new Error("Distribution server build failed")
  }, catch: () => new DistributionBuildFailed() })
  const digest = createHash("sha256").update(yield* fs.readFile(join(output, "server.js"))).digest("hex")
  yield* fs.writeFileString(join(output, "package.json"), `{"name":"@magnitudedev/distribution-server","version":"0.0.0-${digest.slice(0, 12)}","private":true,"type":"module","exports":{".":{"types":"./server.d.ts","import":"./server.js"}},"dependencies":{"pg":"8.20.0"}}\n`)
  yield* fs.writeFileString(join(output, "server.d.ts"), 'import type { Pool } from "pg";\nexport declare function createDistributionServer(input: unknown): Promise<{pool: Pool; check(request: Request, country?: string): Promise<Response>; download(request: Request, country?: string): Promise<Response>; installer(request: Request, country?: string): Promise<Response>}>;\n')
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
