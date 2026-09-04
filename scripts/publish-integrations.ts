import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Schema, Stream } from "effect"
import { resolve } from "node:path"

class IntegrationPublicationFailed extends Schema.TaggedError<IntegrationPublicationFailed>()(
  "IntegrationPublicationFailed", { message: Schema.String },
) {}

const project = resolve(import.meta.dir, "..")
const Manifest = Schema.Struct({ name: Schema.String, version: Schema.String })
const Pack = Schema.Array(Schema.Struct({ filename: Schema.String, integrity: Schema.String }))
const text = <E, R>(stream: Stream.Stream<Uint8Array, E, R>) => stream.pipe(Stream.decodeText(), Stream.runFold("", (text, part) => text + part))
const run = (args: string[], cwd: string) => Effect.scoped(Effect.gen(function* () {
  const child = yield* Command.make("npm", ...args).pipe(Command.workingDirectory(cwd), Command.start)
  const [code, stdout, stderr] = yield* Effect.all([child.exitCode, text(child.stdout), text(child.stderr)], { concurrency: "unbounded" })
  return { code, stdout, stderr }
}))

// Called only by publication workflows, after tarball acceptance. Publishing the
// public contract first makes the pinned companion installable before CLI release.
const program = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const temporary = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-integration-publish-" })
  for (const directory of ["packages/integration-protocol", "integrations/pi"]) {
    const cwd = resolve(project, directory)
    const manifest = yield* fs.readFileString(resolve(cwd, "package.json")).pipe(Effect.flatMap(Schema.decodeUnknown(Schema.parseJson(Manifest))))
    const packed = yield* run(["pack", "--ignore-scripts", "--json", "--pack-destination", temporary], cwd)
    if (packed.code !== 0) return yield* new IntegrationPublicationFailed({ message: packed.stderr })
    const packages = yield* Schema.decodeUnknown(Schema.parseJson(Pack))(packed.stdout)
    if (packages.length !== 1) return yield* new IntegrationPublicationFailed({ message: "Expected one tarball" })
    const artifact = packages[0]!
    const specifier = `${manifest.name}@${manifest.version}`
    const existing = yield* run(["view", specifier, "dist.integrity", "--json"], cwd)
    if (existing.code !== 0) {
      if (!existing.stderr.includes("E404")) return yield* new IntegrationPublicationFailed({ message: existing.stderr })
      const published = yield* run(["publish", resolve(temporary, artifact.filename), "--access", "public"], cwd)
      if (published.code !== 0) return yield* new IntegrationPublicationFailed({ message: published.stderr })
    }
    const verified = yield* run(["view", specifier, "dist.integrity", "--json"], cwd)
    if (verified.code !== 0 || (yield* Schema.decodeUnknown(Schema.parseJson(Schema.String))(verified.stdout)) !== artifact.integrity) {
      return yield* new IntegrationPublicationFailed({ message: `Published integrity differs for ${specifier}. Bump its version; never overwrite an existing release.` })
    }
    yield* Console.log(`Verified ${specifier}`)
  }
}))

if (import.meta.main) BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
