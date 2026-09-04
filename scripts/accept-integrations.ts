import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Schema } from "effect"
import { resolve } from "node:path"

class IntegrationAcceptanceFailed extends Schema.TaggedError<IntegrationAcceptanceFailed>()(
  "IntegrationAcceptanceFailed", { message: Schema.String },
) {}

const project = resolve(import.meta.dir, "..")
const encodeString = Schema.encodeSync(Schema.parseJson(Schema.String))
const Packed = Schema.Array(Schema.Struct({ filename: Schema.String }))
const checked = (executable: string, args: readonly string[], cwd: string, environment: Record<string, string> = {}) => Command.make(executable, ...args).pipe(
  Command.workingDirectory(cwd), Command.env(environment), Command.stdout("inherit"), Command.stderr("inherit"), Command.exitCode,
  Effect.flatMap((code) => code === 0 ? Effect.void : Effect.fail(new IntegrationAcceptanceFailed({ message: `${executable} ${args.join(" ")} failed (${code})` }))),
)

// No workspace symlinks, source imports, or repository node_modules are available
// to the consumer. Exercise both Node and Bun through Pi's own resource loader.
const program = Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-integration-acceptance-" })
  yield* checked("bun", ["run", "build"], resolve(project, "integrations/pi"))
  const tarballs: string[] = []
  for (const directory of ["packages/integration-protocol", "integrations/pi"]) {
    const output = yield* Command.make("npm", "pack", "--ignore-scripts", "--json", "--pack-destination", root).pipe(Command.workingDirectory(resolve(project, directory)), Command.string)
    const packed = yield* Schema.decodeUnknown(Schema.parseJson(Packed))(output)
    if (packed.length !== 1) return yield* new IntegrationAcceptanceFailed({ message: "Expected exactly one package tarball" })
    tarballs.push(resolve(root, packed[0]!.filename))
  }
  yield* fs.writeFileString(resolve(root, "package.json"), '{"private":true,"type":"module"}')
  yield* checked("npm", ["install", "--no-audit", "--no-fund", ...tarballs], root)
  const agentDir = resolve(root, "agent")
  const workspace = resolve(root, "workspace")
  yield* fs.makeDirectory(agentDir)
  yield* fs.makeDirectory(workspace)
  const probe = resolve(root, "probe.mjs")
  yield* fs.writeFileString(probe, `
import assert from 'node:assert/strict'
import { resolve } from 'node:path'
import { DefaultResourceLoader } from '@earendil-works/pi-coding-agent'
const loader = new DefaultResourceLoader({ cwd: ${encodeString(workspace)}, agentDir: ${encodeString(agentDir)}, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true })
await loader.reload()
const result = loader.getExtensions()
assert.deepEqual(result.errors, [])
assert.equal(result.extensions.length, 1)
assert.deepEqual([...result.extensions[0].commands.keys()].sort(), ['load-model', 'stop-model'])
assert.equal(result.runtime.pendingProviderRegistrations[0].name, 'magnitude')
for (const handler of result.extensions[0].handlers.get('session_shutdown') ?? []) await handler({}, {})
console.log('Packed Magnitude extension loaded through Pi with both commands and provider', process.versions.bun ? 'Bun ' + process.versions.bun : 'Node ' + process.version)
`)
  yield* checked("node", [resolve(root, "node_modules/@earendil-works/pi-coding-agent/dist/cli.js"), "install", resolve(root, "node_modules/@magnitudedev/pi")], workspace, { PI_CODING_AGENT_DIR: agentDir })
  // Native install must use the same isolated agent directory as the loader.
  for (const runtime of ["node", "bun"]) yield* checked(runtime, [probe], root)
  yield* Console.log("Integration tarball acceptance passed (Node and Bun).")
}))

if (import.meta.main) BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
