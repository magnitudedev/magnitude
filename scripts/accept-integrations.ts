import * as Command from "@effect/platform/Command"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext, BunRuntime } from "@effect/platform-bun"
import { Console, Effect, Schema } from "effect"
import { resolve } from "node:path"
import {
  readPluginCandidate,
  PluginAcceptanceReceiptSchema,
} from "../packages/release/src/plugin-candidate"

class IntegrationAcceptanceFailed extends Schema.TaggedError<IntegrationAcceptanceFailed>()(
  "IntegrationAcceptanceFailed", { message: Schema.String },
) {}

const encodeString = Schema.encodeSync(Schema.parseJson(Schema.String))
const checked = (
  executable: string,
  args: readonly string[],
  cwd: string,
  environment: Record<string, string> = {}
) =>
  Command.make(executable, ...args).pipe(
    Command.workingDirectory(cwd),
    Command.env(environment),
    Command.stdout("inherit"),
    Command.stderr("inherit"),
    Command.exitCode,
    Effect.flatMap((code) =>
      code === 0
        ? Effect.void
        : Effect.fail(
            new IntegrationAcceptanceFailed({
              message: `${executable} ${args.join(" ")} failed (${code})`,
            })
          )
    )
  )

// No workspace symlinks, source imports, or repository node_modules are available
// to the consumer. Exercise both Node and Bun through Pi's own resource loader.
const program = Effect.scoped(
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const directory = resolve(
      process.argv[2] ?? "release/integration-candidate"
    )
    const candidate = yield* readPluginCandidate(directory)
    const root = yield* fs.makeTempDirectoryScoped({
      prefix: "magnitude-integration-acceptance-",
    })
    yield* fs.writeFileString(
      resolve(root, "package.json"),
      JSON.stringify({
        private: true,
        type: "module",
        dependencies: {
          "@earendil-works/pi-ai": "0.84.4",
          "@earendil-works/pi-coding-agent": "0.84.4",
          "@earendil-works/pi-tui": "0.84.4",
        },
      })
    )
    yield* checked(
      "npm",
      ["install", "--no-audit", "--no-fund", ...candidate.paths],
      root
    )
    const agentDir = resolve(root, "agent")
    const workspace = resolve(root, "workspace")
    yield* fs.makeDirectory(agentDir)
    yield* fs.makeDirectory(workspace)
    const probe = resolve(root, "probe.mjs")
    yield* fs.writeFileString(
      probe,
      `
import assert from 'node:assert/strict'
import { resolve } from 'node:path'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { DefaultResourceLoader, createAgentSession, SessionManager } from '@earendil-works/pi-coding-agent'
const sharedSkills = resolve(${encodeString(workspace)}, process.versions.bun ? 'bun-shared-skills' : 'node-shared-skills')
await mkdir(sharedSkills, { recursive: true })
const loader = new DefaultResourceLoader({ cwd: ${encodeString(workspace)}, agentDir: ${encodeString(
        agentDir
      )}, noSkills: true, additionalSkillPaths: [sharedSkills], noPromptTemplates: true, noThemes: true, noContextFiles: true })
await loader.reload()
const result = loader.getExtensions()
assert.deepEqual(result.errors, [])
assert.equal(result.extensions.length, 1)
assert.deepEqual([...result.extensions[0].commands.keys()].sort(), ['load-model', 'magnitude-setup', 'stop-model'])
assert.equal(result.runtime.pendingProviderRegistrations[0].name, 'magnitude')
const { session } = await createAgentSession({ cwd: ${encodeString(workspace)}, agentDir: ${encodeString(agentDir)}, resourceLoader: loader, sessionManager: SessionManager.inMemory(${encodeString(workspace)}), tools: [] })
await session.bindExtensions({ mode: 'print' })
const skills = loader.getSkills()
assert.equal(skills.skills.filter(skill => skill.name === 'magnitude').length, 1, 'Packed usage skill must be discovered without the CLI')
assert.equal(skills.diagnostics.filter(diagnostic => diagnostic.type === 'collision').length, 0)
const sharedSkill = resolve(sharedSkills, 'magnitude/SKILL.md')
await mkdir(resolve(sharedSkills, 'magnitude'), { recursive: true })
await writeFile(sharedSkill, await readFile(skills.skills.find(skill => skill.name === 'magnitude').filePath))
// The interactive connector installs the shared skill after package-first onboarding.
// Reload must replace the fallback, not retain both copies or report a collision.
for (let reload = 0; reload < 2; reload++) {
  await session.reload()
  const current = loader.getSkills()
  assert.deepEqual(current.skills.filter(skill => skill.name === 'magnitude').map(skill => skill.filePath), [sharedSkill])
  assert.equal(current.diagnostics.filter(diagnostic => diagnostic.type === 'collision').length, 0)
}
if (${process.argv.includes("--daemon")}) {
  const notifications = []
  const ctx = { ui: { notify: (message, level) => notifications.push({ message, level }) } }
  const commands = loader.getExtensions().extensions[0].commands
  const completions = await commands.get('load-model').getArgumentCompletions('')
  assert.ok(Array.isArray(completions), 'Catalog RPC must succeed through the bundled SDK')
  await commands.get('stop-model').handler('', ctx)
  assert.equal(notifications.filter(event => event.level === 'error').length, 0, JSON.stringify(notifications))
  assert.ok(notifications.some(event => event.message === 'Stopped the active Magnitude model.'))
}
// SDK disposal does not emit the host's shutdown event. Exercise that lifecycle
// explicitly so command RPC runtimes release their resources before process exit.
await session.extensionRunner.emit({ type: 'session_shutdown', reason: 'exit' })
session.dispose()
console.log('Packed Magnitude extension loaded through Pi with all commands and provider', process.versions.bun ? 'Bun ' + process.versions.bun : 'Node ' + process.version)
`
    )
    yield* checked(
      "node",
      [
        resolve(
          root,
          "node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
        ),
        "install",
        resolve(root, "node_modules/@magnitudedev/pi-extension"),
      ],
      workspace,
      { PI_CODING_AGENT_DIR: agentDir }
    )
    // Native install must use the same isolated agent directory as the loader.
    for (const runtime of ["node", "bun"])
      yield* checked(runtime, [probe], root)
    // Re-read hashes after execution; acceptance is tied to immutable input bytes.
    const after = yield* readPluginCandidate(directory)
    if (JSON.stringify(candidate.receipt) !== JSON.stringify(after.receipt))
      return yield* new IntegrationAcceptanceFailed({
        message: "Plugin candidate changed during acceptance",
      })
    yield* fs.writeFileString(
      `${directory}/accepted.json`,
      yield* Schema.encode(
        Schema.parseJson(PluginAcceptanceReceiptSchema, { space: 2 })
      )(candidate.receipt)
    )
    yield* Console.log("Integration tarball acceptance passed (Node and Bun).")
  })
)

if (import.meta.main) BunRuntime.runMain(program.pipe(Effect.provide(BunContext.layer)))
