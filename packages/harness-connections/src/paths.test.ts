import { expect, it } from "vitest"
import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect } from "effect"
import { harnessConnectionPaths, resolveHarnessConnectionPaths } from "./paths"

  it("uses resolved configuration roots only outside isolated profiles", () => {
    const environment = { HERMES_HOME: "/resolved/hermes", CODEX_HOME: "/resolved/codex", PI_CODING_AGENT_DIR: "/resolved/pi" }
    expect(harnessConnectionPaths(undefined, environment).hermes).toBe("/resolved/hermes/config.yaml")
    expect(harnessConnectionPaths(undefined, environment).codexUser).toBe("/resolved/codex/config.toml")
    const isolated = harnessConnectionPaths("/isolated", environment)
    expect(isolated.hermes).toBe("/isolated/.hermes/config.yaml")
    expect(isolated.piSettings).toBe("/isolated/.pi/agent/settings.json")
  })

it("uses explicit harness config paths and agent directories", () => {
  const environment = { PI_CODING_AGENT_DIR: "/custom/agent", OPENCODE_CONFIG: "/custom/opencode.jsonc", OPENCLAW_CONFIG_PATH: "/custom/claw.json" }
  const paths = harnessConnectionPaths(undefined, environment)
  expect(paths.ompModels).toBe("/custom/agent/models.yml")
  expect(paths.ompSettings).toBe("/custom/agent/config.yml")
  expect(paths.opencode).toBe("/custom/opencode.jsonc")
  expect(paths.openclaw).toBe("/custom/claw.json")
  expect(harnessConnectionPaths(undefined, { XDG_CONFIG_HOME: "/custom/xdg" }).opencode).toBe("/custom/xdg/opencode/opencode.json")
  const isolated = harnessConnectionPaths("/isolated", environment)
  expect(isolated.ompModels).toBe("/isolated/.omp/agent/models.yml")
  expect(isolated.opencode).toBe("/isolated/.config/opencode/opencode.json")
  expect(isolated.openclaw).toBe("/isolated/.openclaw/openclaw.json")
})

it("keeps Cline model catalogs beside the selected provider settings", () => {
  const root = harnessConnectionPaths(undefined, { CLINE_DATA_DIR: "/custom/cline" })
  expect(root.clineProviders).toBe("/custom/cline/settings/providers.json")
  expect(root.clineModels).toBe("/custom/cline/settings/models.json")
  const file = harnessConnectionPaths(undefined, { CLINE_PROVIDER_SETTINGS_PATH: "/custom/providers.json" })
  expect(file.clineProviders).toBe("/custom/providers.json")
  expect(file.clineModels).toBe("/custom/models.json")
})

it("honors OMP named profiles before an agent-directory override", () => {
  const paths = harnessConnectionPaths(undefined, { OMP_PROFILE: "work", PI_CODING_AGENT_DIR: "/other" })
  expect(paths.ompModels).toMatch(/\/\.omp\/profiles\/work\/agent\/models.yml$/)
  expect(harnessConnectionPaths(undefined, { OMP_PROFILE: "", PI_PROFILE: "work", PI_CODING_AGENT_DIR: "/other" }).ompModels).toBe("/other/models.yml")
  expect(() => harnessConnectionPaths(undefined, { OMP_PROFILE: "../escape" })).toThrow("Invalid Oh My Pi profile name")
})

it("prefers existing global JSONC without creating files", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-config-paths-" })
  const paths = harnessConnectionPaths(root)
  expect((yield* resolveHarnessConnectionPaths(root)).opencode).toBe(paths.opencode)
  expect(yield* fs.readDirectory(root)).toEqual([])
  yield* fs.makeDirectory(`${root}/.config/opencode`, { recursive: true })
  yield* fs.writeFileString(`${paths.opencode}c`, "{}")
  expect((yield* resolveHarnessConnectionPaths(root)).opencode).toBe(`${paths.opencode}c`)
  expect((yield* resolveHarnessConnectionPaths(undefined, { OPENCODE_CONFIG: paths.opencode })).opencode).toBe(paths.opencode)
})).pipe(Effect.provide(BunContext.layer))))
