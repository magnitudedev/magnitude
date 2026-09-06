import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import { HERMES_PLUGIN_FILES, inspectHermesPluginContent } from "@magnitudedev/release/hermes-plugin-content"
import { Effect, Exit, Option, Schema } from "effect"
import { dirname } from "node:path"
import { pathToFileURL } from "node:url"
import { parse } from "yaml"
import { describe, expect, it } from "vitest"
import { connectionTransaction } from "../transaction"
import { makeHermesCompanion } from "./hermes-package"
import { HermesPackageSelectionSchema } from "./hermes-package-state"

const fixture = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-package-test-" })
  const source = `${directory}/source`
  for (const path of HERMES_PLUGIN_FILES) {
    yield* fs.makeDirectory(dirname(`${source}/${path}`), { recursive: true })
    yield* fs.writeFileString(`${source}/${path}`, path === "plugin.yaml"
      ? "name: magnitude\nversion: 0.0.1\n" : `fixture ${path}`)
  }
  const { metadata } = yield* inspectHermesPluginContent(source, MAGNITUDE_RPC_VERSION)
  yield* fs.writeFileString(`${source}/dist/magnitude-plugin.json`, JSON.stringify(metadata))
  const selection = yield* Schema.decodeUnknown(HermesPackageSelectionSchema)({
    source: pathToFileURL(source).href, revision: "a".repeat(40), contentFingerprint: metadata.contentFingerprint,
  })
  const home = `${directory}/home`
  const root = `${home}/plugins/magnitude`
  yield* fs.makeDirectory(home)
  yield* fs.writeFileString(`${home}/config.yaml`, "# preserve this comment\nplugins:\n  enabled: [other]\n  disabled: [magnitude]\nmodel:\n  default: user-model\n")
  const executable = `${directory}/hermes`
  yield* fs.writeFileString(executable, `#!/usr/bin/env bun
import { cp, mkdir, readFile, rm, writeFile, appendFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
const home = process.env.HERMES_HOME
if (!home || !home.includes('magnitude-hermes-package-test-')) throw new Error('Not a fixture home')
const [group, action, source, ...args] = process.argv.slice(2)
if (group !== 'plugins' || args.includes('--force')) throw new Error('Invalid native operation')
const root = home + '/plugins/magnitude'
const sidecar = home + '/plugins/.install-metadata.json'
await appendFile(home + '/commands', JSON.stringify(process.argv.slice(2)) + '\\n')
const records = existsSync(sidecar) ? JSON.parse(await readFile(sidecar, 'utf8')) : {}
if (action === 'install') {
  if (existsSync(root) || !args.includes('--no-enable')) throw new Error('Unsafe install')
  await mkdir(home + '/plugins', { recursive: true })
  await cp(fileURLToPath(source), root, { recursive: true })
  records.magnitude = { source, revision: args[args.indexOf('--ref') + 1], pinned: true }
} else if (action === 'remove' && source === 'magnitude') {
  await rm(root, { recursive: true })
  delete records.magnitude
} else throw new Error('Unsupported action')
await writeFile(sidecar, JSON.stringify(records))
if (existsSync(home + '/fail-' + action)) process.exit(1)
`)
  yield* fs.chmod(executable, 0o755)
  const adapter = makeHermesCompanion({ hermes: `${home}/config.yaml` }, selection)
  const installation = { executable }
  const connect = () => adapter.reconcile({ installation, previous: Option.none() })
  const config = fs.readFileString(`${home}/config.yaml`).pipe(Effect.map(text => ({ text, value: parse(text) })))
  return { fs, directory, source, selection, home, root, adapter, installation, connect, config }
})
const run = <A, E>(test: Effect.Effect<A, E, FileSystem.FileSystem | import("@effect/platform/Path").Path | import("@effect/platform/CommandExecutor").CommandExecutor | import("effect/Scope").Scope>) =>
  Effect.runPromise(Effect.scoped(test).pipe(Effect.provide(BunContext.layer)))

describe("Hermes companion ownership", () => {
  it("installs through the native scanner path, reconnects idempotently, and restores enablement without touching skills", () => run(Effect.gen(function* () {
    const f = yield* fixture
    yield* f.fs.makeDirectory(`${f.home}/skills/magnitude`, { recursive: true })
    yield* f.fs.writeFileString(`${f.home}/skills/magnitude/SKILL.md`, "user skill")
    const first = yield* connectionTransaction(f.connect())
    expect(first.status).toBe("installed")
    expect(first.state.ownership).toBe("magnitude")
    const second = yield* connectionTransaction(f.adapter.reconcile({ installation: f.installation, previous: Option.some(first.state) }))
    expect(second.status).toBe("already-installed")
    const enabled = yield* f.config
    expect(enabled.text).toContain("# preserve this comment")
    expect(enabled.value.plugins).toEqual({ enabled: ["other", "magnitude"], disabled: [] })
    yield* connectionTransaction(f.adapter.disconnect({ installation: f.installation, state: second.state }))
    expect(yield* f.fs.exists(f.root)).toBe(false)
    expect((yield* f.config).value.plugins).toEqual({ enabled: ["other"], disabled: ["magnitude"] })
    expect(yield* f.fs.readFileString(`${f.home}/skills/magnitude/SKILL.md`)).toBe("user skill")
    expect((yield* f.fs.readFileString(`${f.home}/commands`)).trim().split("\n")).toHaveLength(2)
  })))

  it("borrows a verified existing plugin and restores only its enablement", () => run(Effect.gen(function* () {
    const f = yield* fixture
    yield* f.fs.makeDirectory(`${f.home}/plugins`)
    yield* f.fs.copy(f.source, f.root)
    const connected = yield* connectionTransaction(f.connect())
    expect(connected.state.ownership).toBe("pre-existing")
    yield* connectionTransaction(f.adapter.disconnect({ installation: f.installation, state: connected.state }))
    expect(yield* f.fs.exists(f.root)).toBe(true)
    expect(yield* f.fs.exists(`${f.home}/commands`)).toBe(false)
    expect((yield* f.config).value.plugins.disabled).toEqual(["magnitude"])
  })))

  for (const change of ["contents", "extra-file", "symlink", "provenance"]) {
    it(`preserves an owned package with changed ${change}`, () => run(Effect.gen(function* () {
      const f = yield* fixture
      const connected = yield* connectionTransaction(f.connect())
      if (change === "contents") yield* f.fs.writeFileString(`${f.root}/progress.py`, "user change")
      if (change === "extra-file") yield* f.fs.writeFileString(`${f.root}/notes.txt`, "user notes")
      if (change === "symlink") yield* f.fs.symlink(f.source, `${f.root}/user-link`)
      if (change === "provenance") yield* f.fs.writeFileString(`${f.home}/plugins/.install-metadata.json`, "{}")
      const result = yield* Effect.exit(connectionTransaction(f.adapter.disconnect({ installation: f.installation, state: connected.state })))
      expect(Exit.isFailure(result)).toBe(true)
      expect(yield* f.fs.exists(f.root)).toBe(true)
      expect((yield* f.config).value.plugins.enabled).toContain("magnitude")
      expect((yield* f.fs.readFileString(`${f.home}/commands`)).trim().split("\n")).toHaveLength(1)
    })))
  }

  it("compensates a later connection failure, including an install that failed after writing its receipt", () => run(Effect.gen(function* () {
    const f = yield* fixture
    const before = (yield* f.config).text
    const result = yield* Effect.exit(connectionTransaction(f.connect().pipe(Effect.zipRight(Effect.fail("later failure")))))
    expect(Exit.isFailure(result)).toBe(true)
    expect(yield* f.fs.exists(f.root)).toBe(false)
    expect((yield* f.config).text).toBe(before)
    yield* f.fs.writeFileString(`${f.home}/fail-install`, "")
    const partial = yield* Effect.exit(connectionTransaction(f.connect()))
    expect(Exit.isFailure(partial)).toBe(true)
    expect(yield* f.fs.exists(f.root)).toBe(false)
    expect((yield* f.config).text).toBe(before)
  })))

  it("rolls back an owned upgrade to the exact prior source and revision", () => run(Effect.gen(function* () {
    const f = yield* fixture
    const first = yield* connectionTransaction(f.connect())
    const desired = yield* Schema.decodeUnknown(HermesPackageSelectionSchema)({ ...f.selection, revision: "b".repeat(40) })
    const updated = makeHermesCompanion({ hermes: `${f.home}/config.yaml` }, desired)
    const failed = yield* Effect.exit(connectionTransaction(updated.reconcile({ installation: f.installation, previous: Option.some(first.state) })
      .pipe(Effect.zipRight(Effect.fail("later failure")))))
    expect(Exit.isFailure(failed)).toBe(true)
    const restored = yield* connectionTransaction(f.adapter.reconcile({ installation: f.installation, previous: Option.some(first.state) }))
    expect(restored.status).toBe("already-installed")
    expect(restored.state).toEqual(first.state)
  })))
})
