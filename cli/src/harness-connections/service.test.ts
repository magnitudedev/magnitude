import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { HARNESS_PRIORITY, HarnessIdSchema } from "@magnitudedev/client-common"
import { ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { parse } from "jsonc-parser"
import { delimiter, dirname, resolve } from "node:path"
import { parseDocument } from "yaml"
import { describe, expect, it } from "vitest"
import { HarnessModelSchema, type HarnessCompanionPackage } from "./contract"
import { PiPackageEntrySchema } from "./connectors/pi-settings"
import {
  ANTHROPIC_BASE_URL,
  CLAUDE_GATEWAY_DISCOVERY,
  CODEX_PROXY_BASE_URL,
  OPENAI_BASE_URL,
  PI_COMPANION_EXTENSION_PATH,
  PI_COMPANION_PACKAGE_IDENTITY,
  PI_COMPANION_PACKAGE_SOURCE,
  anthropicLocalModelId,
  clineModelCatalog,
  clineModelRegistryEntry,
  clineProviderSettings,
  codexLocalModelId,
  codexModelCatalog,
  harnessExecutableSearchPath,
  hermesProviderConfig,
  hermesReasoningOverrides,
  makeHarnessConnectionService,
  makeHarnessConnectorRegistry,
  ohMyPiProviderConfig,
  openClawAgentConfig,
  openClawProviderConfig,
  openCodeProviderConfig,
  piPackageExtensionEnabled,
  piProviderConfig,
  updateJsonc,
  updateYaml,
  type HarnessConnectionPaths,
} from "./service"

const model = ProviderModelIdSchema.make("local/model")
const secondModel = ProviderModelIdSchema.make("local/second-model")
const high = ReasoningEffortSchema.make("high")
const none = ReasoningEffortSchema.make("none")
const low = ReasoningEffortSchema.make("low")
const xhigh = ReasoningEffortSchema.make("xhigh")
const adaptive = ReasoningEffortSchema.make("adaptive")
const deliberate = ReasoningEffortSchema.make("deliberate")
const models = [
  {
    id: model,
    name: "Local Model (Q4)",
    description: "A local test model.",
    contextWindow: 50_000,
    maxOutputTokens: 32_768,
    capabilities: {
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: { supported: true, efforts: [high], defaultEffort: high },
    },
  },
  {
    id: secondModel,
    name: "Second Model (Q6)",
    description: "Another local test model.",
    contextWindow: 32_768,
    maxOutputTokens: 32_768,
    capabilities: {
      vision: true,
      tools: true,
      structuredOutput: true,
      reasoning: { supported: false, efforts: [] },
    },
  },
] as const

const bundledCodexCatalog = {
  future_root_field: { preserved: true },
  models: [{
    slug: "gpt-openai-fixture",
    display_name: "OpenAI Fixture",
    visibility: "list",
    unknown_future_field: { preserved: true },
  }],
}

const fixturePaths = (root: string): HarnessConnectionPaths => ({
  manifest: `${root}/magnitude/harness-connections.json`,
  piModels: `${root}/pi/models.json`,
  piSettings: `${root}/pi/settings.json`,
  opencode: `${root}/opencode/opencode.json`,
  hermes: `${root}/hermes/config.yaml`,
  openclaw: `${root}/openclaw/openclaw.json`,
  codex: `${root}/codex/magnitude.config.toml`,
  codexUser: `${root}/codex/config.toml`,
  codexModels: `${root}/codex/magnitude.models.json`,
  claude: `${root}/claude/settings.json`,
  ompModels: `${root}/omp/models.yml`,
  ompSettings: `${root}/omp/config.yml`,
  clineProviders: `${root}/cline/providers.json`,
  clineModels: `${root}/cline/models.json`,
  skillInstallations: {
    "shared-agents": {
      skillFile: `${root}/skills/agents/magnitude/SKILL.md`,
    },
    "hermes-user": {
      skillFile: `${root}/skills/hermes/magnitude/SKILL.md`,
    },
    "claude-user": {
      skillFile: `${root}/skills/claude/magnitude/SKILL.md`,
    },
    "cline-user": {
      skillFile: `${root}/skills/cline/magnitude/SKILL.md`,
    },
  },
})

const stringifyJson = Schema.encodeSync(Schema.parseJson(Schema.Unknown, { space: 2 }))
const readJson = (source: string): unknown => parse(source)
const readYaml = (source: string): unknown => parseDocument(source).toJS()

const testPiCompanion: HarnessCompanionPackage = {
  description: {
    name: "Magnitude for Pi",
    source: "npm:@magnitudedev/pi@0.0.1",
    securityNotice: "Pi extensions execute with your user permissions.",
  },
  activationInstructions: Option.some("Restart existing Pi sessions or run /reload to activate the extension."),
  reconcile: ({ previous }) => Effect.succeed({
    state: Option.getOrElse(previous, () => ({
      identity: "@magnitudedev/pi",
      source: PI_COMPANION_PACKAGE_SOURCE,
      ownership: "magnitude" as const,
    })),
    status: Option.isSome(previous) ? "already-installed" as const : "installed" as const,
  }),
  disconnect: () => Effect.void,
}

const installedService = (paths: HarnessConnectionPaths, resolvedModels = models as ReadonlyArray<(typeof models)[number]>) =>
  makeHarnessConnectionService({
    paths,
    registry: makeHarnessConnectorRegistry(paths, {
      readCodexBundledCatalog: () => Effect.succeed(bundledCodexCatalog),
      piCompanion: testPiCompanion,
    }),
    detect: (connector) => Effect.succeed(Option.some({ executable: `/installed/${connector.id}` })),
    resolveModels: Effect.succeed(resolvedModels),
    now: () => new Date("2026-08-26T00:00:00.000Z"),
    installStartup: Effect.void,
  })

const initialFiles = (paths: HarnessConnectionPaths): Readonly<Record<string, string>> => ({
  [paths.piModels]: stringifyJson({ theme: "dark", providers: {} }),
  [paths.piSettings]: stringifyJson({ theme: "dark", defaultProvider: "user", defaultModel: "user/model" }),
  [paths.opencode]: stringifyJson({ theme: "dark", provider: {}, model: "user/model" }),
  [paths.hermes]: "theme: dark\nmodel:\n  provider: user\n  default: model\n",
  [paths.openclaw]: stringifyJson({
    theme: "dark",
    models: { providers: {} },
    agents: {
      defaults: { model: { primary: "user/model" } },
      list: [{ id: "main", model: "user/model" }],
    },
  }),
  [paths.codexUser]: 'model_provider = "openai"\nmodel = "user/model"\n',
  [paths.claude]: stringifyJson({ theme: "dark", model: "user/model", env: { USER_KEY: "preserve" } }),
  [paths.ompModels]: "theme: dark\n",
  [paths.ompSettings]: "theme: dark\nmodelRoles:\n  default: user/model\n",
  [paths.clineProviders]: stringifyJson({
    version: 1,
    modes: { keep: true },
    providers: { "user-provider": { settings: { provider: "user-provider", model: "user/model" } } },
    lastUsedProvider: "user-provider",
  }),
  [paths.clineModels]: stringifyJson({ version: 1, providers: {} }),
})

const writeFixtures = (files: Readonly<Record<string, string>>) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  for (const [file, contents] of Object.entries(files)) {
    yield* fs.makeDirectory(file.slice(0, file.lastIndexOf("/")), { recursive: true })
    yield* fs.writeFileString(file, contents.endsWith("\n") ? contents : `${contents}\n`)
  }
})

const writeFakePiExecutable = (
  paths: HarnessConnectionPaths,
  failCommand?: "install" | "remove" | "install-after" | "remove-after",
  version = "0.83.0",
) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const executable = `${paths.piSettings.slice(0, paths.piSettings.lastIndexOf("/"))}/pi-test`
  const encodedFailCommand = failCommand === undefined ? "undefined" : stringifyJson(failCommand)
const source = `#!/usr/bin/env bun
import { dirname, isAbsolute, relative, resolve } from "node:path"
const settingsPath = new URL("./settings.json", import.meta.url).pathname
const logPath = new URL("./package-commands.jsonl", import.meta.url).pathname
const [command, source] = process.argv.slice(2)
if (command === "--version") { console.log(${JSON.stringify(version)}); process.exit(0) }
if (command === ${encodedFailCommand}) process.exit(9)
const settings = await Bun.file(settingsPath).exists() ? await Bun.file(settingsPath).json() : {}
const packages = Array.isArray(settings.packages) ? settings.packages : []
const storedSource = isAbsolute(source) ? relative(dirname(settingsPath), source) : source
const equalSource = (value) => value === source || (!value.startsWith("npm:") && resolve(dirname(settingsPath), value) === resolve(dirname(settingsPath), source))
if (command === "install") {
  settings.packages = packages.some((entry) => equalSource(typeof entry === "string" ? entry : entry.source)) ? packages : [...packages, storedSource]
  const root = source.startsWith("npm:") ? resolve(dirname(settingsPath), "npm/node_modules/@magnitudedev/pi") : source
  await Bun.write(resolve(root, "package.json"), JSON.stringify({ name: "@magnitudedev/pi", version: "0.0.1", pi: { extensions: ["./dist/magnitude.js"] } }))
  await Bun.write(resolve(root, "dist/magnitude.js"), "export default () => {}")
}
else if (command === "remove") settings.packages = packages.filter((entry) => {
  const value = typeof entry === "string" ? entry : entry?.source
  return typeof value !== "string" || !equalSource(value)
})
else process.exit(2)
await Bun.write(settingsPath, JSON.stringify(settings, null, 2) + "\\n")
const previous = await Bun.file(logPath).exists() ? await Bun.file(logPath).text() : ""
await Bun.write(logPath, previous + JSON.stringify({ command, source }) + "\\n")
if (command + "-after" === ${encodedFailCommand}) { console.error("failed after native mutation"); process.exit(9) }
`
  yield* fs.writeFileString(executable, source)
  yield* fs.chmod(executable, 0o755)
  return executable
})

describe("HarnessModel persistence", () => {
  it("derives the 32K-bounded output ceiling for existing descriptors", () => {
    const { maxOutputTokens: _, ...persisted } = models[0]

    const decoded = Schema.decodeUnknownSync(HarnessModelSchema)(persisted)

    expect(decoded.maxOutputTokens).toBe(32_768)
  })
})

describe("HarnessConnector contract and registry", () => {
  it("composes one complete connector module per harness in canonical order", () => {
    const registry = makeHarnessConnectorRegistry(fixturePaths("/tmp/contract"))
    expect(registry.ordered.map(({ id }) => id)).toEqual(HARNESS_PRIORITY)
    for (const connector of registry.ordered) {
      expect(connector.name.length).toBeGreaterThan(0)
      expect(typeof connector.detect).toBe("function")
      expect(typeof connector.connect).toBe("function")
      expect(typeof connector.disconnect).toBe("function")
      expect(typeof connector.launch).toBe("function")
      expect(connector.skillInstallationTarget.length).toBeGreaterThan(0)
    }
  })

  it("maps interoperable harnesses to one shared skill target", () => {
    const registry = makeHarnessConnectorRegistry(fixturePaths("/tmp/skill-targets"))
    expect(Object.fromEntries(registry.ordered.map(({ id, skillInstallationTarget }) => [
      id,
      skillInstallationTarget,
    ]))).toEqual({
      magnitude: "shared-agents",
      pi: "shared-agents",
      opencode: "shared-agents",
      hermes: "hermes-user",
      openclaw: "shared-agents",
      codex: "shared-agents",
      "claude-code": "claude-user",
      "oh-my-pi": "shared-agents",
      cline: "cline-user",
    })
  })

  it("does not detect repository dependencies as installed harnesses", () => {
    expect(harnessExecutableSearchPath([
      "/workspace/node_modules/.bin", "/Users/developer/node_modules/.bin",
      "/Users/developer/.local/bin", "/usr/bin",
    ].join(delimiter))).toBe(["/Users/developer/.local/bin", "/usr/bin"].join(delimiter))
  })

  it("uses all models in model-enumerating provider formats", () => {
    expect(piProviderConfig(models).models.map(({ id }) => id)).toEqual([model, secondModel])
    expect(openCodeProviderConfig(models).models).toMatchObject({
      [model]: { name: "Local Model (Q4)", variants: { high: { reasoningEffort: "high" } } },
      [secondModel]: { name: "Second Model (Q6)", variants: {} },
    })
    expect(openClawProviderConfig(models).models.map(({ id }) => id)).toEqual([model, secondModel])
    expect(ohMyPiProviderConfig(models).models.map(({ id }) => id)).toEqual([model, secondModel])
    expect(clineModelCatalog(models)).toEqual({
      [model]: expect.objectContaining({ id: model, name: "Local Model (Q4)", contextWindow: 50_000 }),
      [secondModel]: expect.objectContaining({ id: secondModel, name: "Second Model (Q6)", contextWindow: 32_768 }),
    })
    expect(hermesProviderConfig().transport).toBe("chat_completions")
    expect(clineProviderSettings(Option.none())).not.toHaveProperty("model")
    expect(clineProviderSettings(Option.some(model))).toHaveProperty("model", model)
  })

  it("keeps protocol assignments explicit", () => {
    expect(OPENAI_BASE_URL).toBe("http://127.0.0.1:10100/inference/v1")
    expect(CODEX_PROXY_BASE_URL).toBe("http://127.0.0.1:10100/inference/v1/proxies/codex")
    expect(ANTHROPIC_BASE_URL).toBe(
      "http://127.0.0.1:10100/inference/anthropic/proxies/claude-code",
    )
    expect(piProviderConfig(models).api).toBe("openai-completions")
    expect(openCodeProviderConfig(models).npm).toBe("@ai-sdk/openai-compatible")
    expect(openClawProviderConfig(models).api).toBe("openai-completions")
    expect(ohMyPiProviderConfig(models).api).toBe("openai-completions")
  })

  it("preserves unrelated JSONC while applying managed keys", () => {
    const source = `{ // user comment\n  "theme": "dark"\n}\n`
    const result = updateJsonc(source, [[["provider", "magnitude"], { api: "openai-completions" }]])
    expect(result).toContain("// user comment")
    expect(readJson(result)).toMatchObject({ theme: "dark", provider: { magnitude: { api: "openai-completions" } } })
  })

  it("projects exact model-relative reasoning domains without changing harness globals", () => {
    const piModels = piProviderConfig(models).models
    expect(piModels[0]).toMatchObject({
      reasoning: true,
      thinkingLevelMap: { off: null, medium: null, high: "high" },
      compat: { supportsReasoningEffort: true },
    })
    expect(piModels[1]).toMatchObject({ reasoning: false })

    expect(ohMyPiProviderConfig(models).models[0]).toMatchObject({
      reasoning: true,
      thinking: {
        efforts: ["high"],
        defaultLevel: "high",
        effortMap: { high: "high" },
        requiresEffort: true,
      },
    })
    expect(openClawProviderConfig(models).models[0]).toMatchObject({
      reasoning: true,
      thinkingLevelMap: { medium: null, high: "high" },
      compat: { supportedReasoningEfforts: ["high"] },
    })
    expect(clineModelCatalog(models)[model]).toMatchObject({
      reasoningOptions: [{ type: "effort", values: ["default", "high"] }],
    })
  })

  it("keeps harness controls distinct from named model efforts", () => {
    const adaptiveModel = {
      ...models[0],
      capabilities: {
        ...models[0].capabilities,
        reasoning: { supported: true, efforts: [adaptive, high], defaultEffort: adaptive },
      },
    } as const
    const namedOnlyModel = {
      ...models[0],
      capabilities: {
        ...models[0].capabilities,
        reasoning: { supported: true, efforts: [deliberate], defaultEffort: deliberate },
      },
    } as const

    expect(piProviderConfig([adaptiveModel]).models[0]?.thinkingLevelMap).toMatchObject({
      medium: "adaptive",
      high: "high",
    })
    expect(openClawProviderConfig([adaptiveModel]).models[0]?.thinkingLevelMap).toMatchObject({
      medium: "adaptive",
      high: "high",
    })
    expect(openClawAgentConfig(adaptiveModel)).toEqual({
      id: "magnitude",
      model: `magnitude/${model}`,
      thinkingDefault: "medium",
    })
    expect(ohMyPiProviderConfig([adaptiveModel]).models[0]?.thinking).toMatchObject({
      efforts: ["medium", "high"],
      defaultLevel: "medium",
      effortMap: { medium: "adaptive", high: "high" },
    })
    expect(clineModelCatalog([adaptiveModel])[model]).toMatchObject({
      reasoningOptions: [{ type: "effort", values: ["default", "high"] }],
    })

    expect(piProviderConfig([namedOnlyModel]).models[0]?.thinkingLevelMap).toMatchObject({ high: "deliberate" })
    expect(openClawAgentConfig(namedOnlyModel).thinkingDefault).toBe("high")
    expect(ohMyPiProviderConfig([namedOnlyModel]).models[0]?.thinking).toMatchObject({
      efforts: ["high"],
      defaultLevel: "high",
      effortMap: { high: "deliberate" },
    })
  })

  it("projects disabled defaults and sparse effort domains without filling holes", () => {
    const toggleModel = {
      ...models[0],
      capabilities: {
        ...models[0].capabilities,
        reasoning: { supported: true, efforts: [none, high], defaultEffort: none },
      },
    } as const
    const sparseModel = {
      ...models[0],
      capabilities: {
        ...models[0].capabilities,
        reasoning: { supported: true, efforts: [none, low, xhigh], defaultEffort: xhigh },
      },
    } as const

    expect(piProviderConfig([sparseModel]).models[0]?.thinkingLevelMap).toEqual({
      off: "none",
      minimal: null,
      low: "low",
      medium: null,
      high: null,
      xhigh: "xhigh",
      max: null,
    })
    expect(openClawAgentConfig(toggleModel).thinkingDefault).toBe("off")
    expect(openClawAgentConfig(sparseModel).thinkingDefault).toBe("xhigh")
    expect(ohMyPiProviderConfig([toggleModel]).models[0]?.thinking).toEqual({
      mode: "effort",
      efforts: ["high"],
      effortMap: { high: "high" },
      requiresEffort: false,
    })
    expect(clineModelCatalog([sparseModel])[model]).toMatchObject({
      reasoningOptions: [
        { type: "toggle" },
        { type: "effort", values: ["default", "low", "xhigh"] },
      ],
    })
  })
})

describe("Pi companion package lifecycle", () => {
  it.each([
    '{"packages":{}}',
    '{"packages":[{"source":"npm:@magnitudedev/pi@0.0.1","extensions":[42]}]}',
    '{"packages":[{"source":"npm:@magnitudedev/pi@0.0.1","autoload":"false"}]}',
    '{"packages":[{"extensions":[]}]}',
  ])("rejects invalid settings before native package operations: %s", async (settings) => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-invalid-settings-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({ ...initialFiles(paths), [paths.piSettings]: settings })
      const beforeSettings = yield* fs.readFileString(paths.piSettings)
      const beforeModels = yield* fs.readFileString(paths.piModels)
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({ paths, detect: () => Effect.succeed(Option.some({ executable })), resolveModels: Effect.succeed(models), installStartup: Effect.void })
      const exit = yield* Effect.exit(service.connect(HarnessIdSchema.make("pi"), { model: Option.none() }))
      expect(String(exit)).toContain("Invalid Pi settings")
      expect(String(exit)).toContain(paths.piSettings)
      expect(yield* fs.readFileString(paths.piSettings)).toBe(beforeSettings)
      expect(yield* fs.readFileString(paths.piModels)).toBe(beforeModels)
      expect(yield* fs.exists(`${root}/pi/package-commands.jsonl`)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("preserves JSONC comments and unrelated fields while enabling and restoring a borrowed package", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-jsonc-" })
      const paths = fixturePaths(root)
      const settings = `{
  // Preserve the user's theme.
  "theme": "dark",
  "packages": [{
    "source": "${PI_COMPANION_PACKAGE_SOURCE}",
    // This package also supplies prompts.
    "prompts": ["./prompts"],
    "future": {"preserve": true},
    "autoload": false,
  }],
}`
      yield* writeFixtures({
        ...initialFiles(paths),
        [paths.piSettings]: settings,
        [`${root}/pi/npm/node_modules/@magnitudedev/pi/package.json`]: stringifyJson({ name: "@magnitudedev/pi", version: "0.0.1", pi: { extensions: ["./dist/magnitude.js"] } }),
        [`${root}/pi/npm/node_modules/@magnitudedev/pi/dist/magnitude.js`]: "export default () => {}",
      })
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({ paths, detect: () => Effect.succeed(Option.some({ executable })), resolveModels: Effect.succeed(models), installStartup: Effect.void })
      yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      const connected = yield* fs.readFileString(paths.piSettings)
      expect(connected).toContain("// Preserve the user's theme.")
      expect(connected).toContain("// This package also supplies prompts.")
      expect(readJson(connected)).toMatchObject({ packages: [{ extensions: [`+${PI_COMPANION_EXTENSION_PATH}`], future: { preserve: true } }] })
      yield* service.disconnect(HarnessIdSchema.make("pi"))
      const restored = yield* fs.readFileString(paths.piSettings)
      expect(restored).toContain("// Preserve the user's theme.")
      expect(restored).toContain("// This package also supplies prompts.")
      expect(readJson(restored)).toEqual(readJson(settings))
      expect(yield* fs.exists(`${root}/pi/package-commands.jsonl`)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it.each([
    ["0.82.1", false],
    ["0.83.0", true],
    ["0.84.4", true],
    ["0.85.0", true],
    ["0.83.0-beta.1", false],
    ["invalid", false],
  ] as const)("checks the minimum Pi version: %s", async (version, supported) => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-version-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths, undefined, version)
      const service = yield* makeHarnessConnectionService({ paths, detect: () => Effect.succeed(Option.some({ executable })), resolveModels: Effect.succeed(models), installStartup: Effect.void })
      const exit = yield* Effect.exit(service.connect(HarnessIdSchema.make("pi"), { model: Option.none() }))
      expect(exit._tag).toBe(supported ? "Success" : "Failure")
      if (!supported) {
        expect(String(exit)).toContain("requires Pi 0.83.0 or newer")
        expect(yield* fs.exists(`${root}/pi/package-commands.jsonl`)).toBe(false)
      }
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it.each(["install-after", "remove-after"] as const)("compensates partial native failure: %s", async (failure) => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-partial-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths, failure)
      const service = yield* makeHarnessConnectionService({ paths, detect: () => Effect.succeed(Option.some({ executable })), resolveModels: Effect.succeed(models), installStartup: Effect.void })
      if (failure === "remove-after") yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      const before = yield* fs.readFileString(paths.piSettings)
      const manifest = yield* fs.readFileString(paths.manifest)
      const exit = yield* Effect.exit(failure === "install-after" ? service.connect(HarnessIdSchema.make("pi"), { model: Option.none() }) : service.disconnect(HarnessIdSchema.make("pi")))
      expect(String(exit)).toContain("failed after native mutation")
      expect(yield* fs.readFileString(paths.piSettings)).toBe(before)
      expect(yield* fs.readFileString(paths.manifest)).toBe(manifest)
      const log = (yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n").map((line) => JSON.parse(line).command)
      expect(log).toEqual(failure === "install-after" ? ["install", "remove"] : ["install", "remove", "install"])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })
  it.each(["unrelated", "filters", "removed"])("preserves subsequent borrowed-package edits: %s", async (change) => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-borrowed-edit-" })
      const paths = fixturePaths(root)
      const existing = { source: PI_COMPANION_PACKAGE_SOURCE, extensions: [], prompts: ["original"], autoload: false }
      yield* writeFixtures({
        ...initialFiles(paths),
        [paths.piSettings]: stringifyJson({ theme: "original", packages: [existing] }),
        [`${root}/pi/npm/node_modules/@magnitudedev/pi/package.json`]: stringifyJson({ name: "@magnitudedev/pi", version: "0.0.1", pi: { extensions: ["./dist/magnitude.js"] } }),
        [`${root}/pi/npm/node_modules/@magnitudedev/pi/dist/magnitude.js`]: "export default () => {}",
      })
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({ paths, detect: () => Effect.succeed(Option.some({ executable })), resolveModels: Effect.succeed(models), installStartup: Effect.void })
      const connected = yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      expect(Option.getOrThrow(connected.companion).status).toBe("enabled")
      const settings = readJson(yield* fs.readFileString(paths.piSettings)) as { packages: Record<string, unknown>[] }
      const edited = { ...settings.packages[0], prompts: ["user changed"], ...(change === "filters" ? { extensions: ["user/*.js"] } : {}) }
      yield* fs.writeFileString(paths.piSettings, stringifyJson({ ...settings, theme: "user changed", packages: change === "removed" ? [] : [edited] }))
      yield* service.disconnect(HarnessIdSchema.make("pi"))
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({ theme: "user changed", packages: change === "removed" ? [] : [{ ...edited, extensions: change === "filters" ? ["user/*.js"] : [] }] })
      expect(yield* fs.exists(`${root}/pi/package-commands.jsonl`)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("repairs missing physical files even when a package remains configured", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-missing-files-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({ paths, detect: () => Effect.succeed(Option.some({ executable })), resolveModels: Effect.succeed(models), installStartup: Effect.void })
      yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      const entrypoint = `${root}/pi/npm/node_modules/@magnitudedev/pi/dist/magnitude.js`
      yield* fs.remove(entrypoint)
      yield* service.sync(HarnessIdSchema.make("pi"))
      expect(yield* fs.exists(entrypoint)).toBe(true)
      expect((yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n")).toHaveLength(2)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("rejects an incompatible borrowed package without replacing user-owned state", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-incompatible-" })
      const paths = fixturePaths(root)
      const settings = stringifyJson({ packages: ["npm:@magnitudedev/pi@9.0.0"] })
      yield* writeFixtures({ ...initialFiles(paths), [paths.piSettings]: settings,
        [`${root}/pi/npm/node_modules/@magnitudedev/pi/package.json`]: stringifyJson({ name: "@magnitudedev/pi", version: "9.0.0", pi: { extensions: ["./dist/magnitude.js"] } }),
      })
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({ paths, detect: () => Effect.succeed(Option.some({ executable })), resolveModels: Effect.succeed(models), installStartup: Effect.void })
      const exit = yield* Effect.exit(service.connect(HarnessIdSchema.make("pi"), { model: Option.none() }))
      expect(String(exit)).toContain("Unsupported Magnitude for Pi package")
      expect(yield* fs.readFileString(paths.piSettings)).toBe(settings + "\n")
      expect(yield* fs.exists(`${root}/pi/package-commands.jsonl`)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("matches Pi's include, exclude, and exact override precedence", () => {
    const enabled = (extensions?: readonly string[]) => piPackageExtensionEnabled(Schema.decodeUnknownSync(PiPackageEntrySchema)({
      source: PI_COMPANION_PACKAGE_SOURCE,
      ...(extensions === undefined ? {} : { extensions }),
    }))
    expect(piPackageExtensionEnabled(PI_COMPANION_PACKAGE_SOURCE)).toBe(true)
    expect(enabled()).toBe(true)
    expect(enabled(["dist/*.js"])).toBe(true)
    expect(enabled(["extensions/other.ts"])).toBe(false)
    expect(enabled(["!dist/*.js"])).toBe(false)
    expect(enabled(["!dist/*.js", "+./dist/magnitude.js"])).toBe(true)
    expect(enabled(["+dist/magnitude.js", "-./dist/magnitude.js"])).toBe(false)
  })

  it("keeps the exact install source synchronized with the publishable package manifest", async () => {
    const manifest = await Bun.file(new URL("../../../integrations/pi/package.json", import.meta.url)).json() as {
      readonly name: string
      readonly version: string
    }

    expect(manifest.name).toBe(PI_COMPANION_PACKAGE_IDENTITY)
    expect(PI_COMPANION_PACKAGE_SOURCE).toBe(`npm:${manifest.name}@${manifest.version}`)
  })

  it("installs, records ownership, reconciles, and removes an owned package through the shared service", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-owned-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({
        paths,
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })

      const result = yield* service.connect(HarnessIdSchema.make("pi"), {
        model: Option.some(model),
        installSkill: true,
      })
      const companion = Option.getOrThrow(result.companion)
      expect(companion).toMatchObject({
        source: PI_COMPANION_PACKAGE_SOURCE,
        status: "installed",
        activationInstructions: Option.some("Restart existing Pi sessions or run /reload to activate the extension."),
      })
      expect(result.skillInstalled).toBe(true)
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        packages: [PI_COMPANION_PACKAGE_SOURCE],
        defaultProvider: "magnitude",
        defaultModel: model,
      })
      expect(readJson(yield* fs.readFileString(paths.manifest))).toMatchObject({
        connections: [{
          harness: "pi",
          companion: {
            identity: PI_COMPANION_PACKAGE_IDENTITY,
            source: PI_COMPANION_PACKAGE_SOURCE,
            ownership: "magnitude",
          },
        }],
      })

      const repeated = yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.some(secondModel) })
      expect(Option.getOrThrow(repeated.companion).status).toBe("already-installed")
      yield* service.disconnect(HarnessIdSchema.make("pi"))
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        packages: [],
        defaultProvider: "user",
        defaultModel: "user/model",
      })
      expect(readJson(yield* fs.readFileString(paths.piModels))).toEqual({ theme: "dark", providers: {} })
      expect((yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n").map((line) => JSON.parse(line)))
        .toEqual([
          { command: "install", source: PI_COMPANION_PACKAGE_SOURCE },
          { command: "remove", source: PI_COMPANION_PACKAGE_SOURCE },
        ])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("tracks a local development package source and always installs the Pi agent skill", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-local-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths)
      const localSource = `${root}/checkout/integrations/pi`
      const service = yield* makeHarnessConnectionService({
        paths,
        registry: makeHarnessConnectorRegistry(paths, { piCompanionSource: localSource }),
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })

      const result = yield* service.connect(HarnessIdSchema.make("pi"), {
        model: Option.some(model),
      })

      expect(result.skillInstalled).toBe(true)
      expect(Option.getOrThrow(result.companion)).toMatchObject({
        source: localSource,
        status: "installed",
      })
      const installedPackages = (readJson(yield* fs.readFileString(paths.piSettings)) as { packages: string[] }).packages
      expect(installedPackages).toHaveLength(1)
      expect(resolve(dirname(paths.piSettings), installedPackages[0]!)).toBe(localSource)
      expect(yield* fs.readFileString(paths.skillInstallations["shared-agents"].skillFile))
        .toContain("name: magnitude")
      expect(readJson(yield* fs.readFileString(paths.manifest))).toMatchObject({
        connections: [{
          harness: "pi",
          companion: { source: localSource, ownership: "magnitude" },
        }],
      })

      yield* fs.remove(paths.skillInstallations["shared-agents"].skillFile)
      yield* service.sync(HarnessIdSchema.make("pi"))
      expect(yield* fs.readFileString(paths.skillInstallations["shared-agents"].skillFile))
        .toContain("name: magnitude")
      yield* service.disconnect(HarnessIdSchema.make("pi"))

      expect((yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n").map((line) => JSON.parse(line)))
        .toEqual([
          { command: "install", source: localSource },
          { command: "remove", source: localSource },
        ])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("replaces a Magnitude-owned companion when the desired package source changes", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-source-change-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths)
      const makeService = (source: string) => makeHarnessConnectionService({
        paths,
        registry: makeHarnessConnectorRegistry(paths, { piCompanionSource: source }),
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })
      const firstSource = `${root}/checkout/integrations/pi-a`
      const secondSource = `${root}/checkout/integrations/pi-b`

      const first = yield* makeService(firstSource)
      yield* first.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      const second = yield* makeService(secondSource)
      yield* second.sync(HarnessIdSchema.make("pi"))

      const installedPackages = (readJson(yield* fs.readFileString(paths.piSettings)) as { packages: string[] }).packages
      expect(installedPackages).toHaveLength(1)
      expect(resolve(dirname(paths.piSettings), installedPackages[0]!)).toBe(secondSource)
      expect(readJson(yield* fs.readFileString(paths.manifest))).toMatchObject({
        connections: [{ companion: { source: secondSource, ownership: "magnitude" } }],
      })
      expect((yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n").map((line) => JSON.parse(line)))
        .toEqual([
          { command: "install", source: firstSource },
          { command: "remove", source: firstSource },
          { command: "install", source: secondSource },
        ])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("minimally enables and exactly restores a pre-existing package without package-manager calls", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-existing-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({
        [`${root}/pi/npm/node_modules/@magnitudedev/pi/package.json`]: stringifyJson({ name: "@magnitudedev/pi", version: "0.0.1", pi: { extensions: ["./dist/magnitude.js"] } }),
        [`${root}/pi/npm/node_modules/@magnitudedev/pi/dist/magnitude.js`]: "export default () => {}",
      })
      const existing = {
        source: PI_COMPANION_PACKAGE_SOURCE,
        extensions: ["extensions/other.ts", `-${PI_COMPANION_EXTENSION_PATH}`],
        custom: { preserve: true },
      }
      yield* writeFixtures({
        ...initialFiles(paths),
        [paths.piSettings]: stringifyJson({
          theme: "dark",
          defaultProvider: "user",
          defaultModel: "user/model",
          packages: [existing],
        }),
      })
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({
        paths,
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })

      const result = yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      expect(Option.getOrThrow(result.companion).status).toBe("enabled")
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        packages: [{
          source: existing.source,
          extensions: ["extensions/other.ts", `+${PI_COMPANION_EXTENSION_PATH}`],
          custom: { preserve: true },
        }],
      })
      expect(yield* fs.exists(`${root}/pi/package-commands.jsonl`)).toBe(false)

      yield* service.disconnect(HarnessIdSchema.make("pi"))
      expect((readJson(yield* fs.readFileString(paths.piSettings)) as { packages: unknown[] }).packages)
        .toEqual([existing])
      expect(yield* fs.exists(`${root}/pi/package-commands.jsonl`)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("reinstalls a missing owned package during sync", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-sync-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths)
      const service = yield* makeHarnessConnectionService({
        paths,
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })
      yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      const settings = readJson(yield* fs.readFileString(paths.piSettings)) as Record<string, unknown>
      yield* fs.writeFileString(paths.piSettings, stringifyJson({ ...settings, packages: [] }))

      yield* service.sync(HarnessIdSchema.make("pi"))

      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        packages: [PI_COMPANION_PACKAGE_SOURCE],
      })
      expect((yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n").map((line) => JSON.parse(line)))
        .toEqual([
          { command: "install", source: PI_COMPANION_PACKAGE_SOURCE },
          { command: "install", source: PI_COMPANION_PACKAGE_SOURCE },
        ])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("compensates package-manager and file effects when the enclosing connection fails", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-rollback-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const executable = yield* writeFakePiExecutable(paths)
      const base = makeHarnessConnectorRegistry(paths)
      const ordered = base.ordered.map((connector) => connector.id === "pi"
        ? { ...connector, connect: () => Effect.die("fixture connect failure") }
        : connector)
      const registry = {
        ordered,
        get: (harness: typeof ordered[number]["id"]) => ordered.find(({ id }) => id === harness)!,
      }
      const service = yield* makeHarnessConnectionService({
        paths,
        registry,
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })

      const exit = yield* Effect.exit(service.connect(HarnessIdSchema.make("pi"), {
        model: Option.some(model),
        installSkill: true,
      }))

      expect(exit._tag).toBe("Failure")
      expect(yield* fs.readFileString(paths.piSettings)).toBe(`${initial[paths.piSettings]}\n`)
      expect(yield* fs.readFileString(paths.piModels)).toBe(`${initial[paths.piModels]}\n`)
      expect(yield* fs.exists(paths.skillInstallations["shared-agents"].skillFile)).toBe(false)
      expect((yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n").map((line) => JSON.parse(line)))
        .toEqual([
          { command: "install", source: PI_COMPANION_PACKAGE_SOURCE },
          { command: "remove", source: PI_COMPANION_PACKAGE_SOURCE },
        ])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("reinstalls an owned package and restores files when disconnect fails", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-disconnect-rollback-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths)
      const base = makeHarnessConnectorRegistry(paths)
      const ordered = base.ordered.map((connector) => connector.id === "pi"
        ? { ...connector, disconnect: () => Effect.die("fixture disconnect failure") }
        : connector)
      const registry = {
        ordered,
        get: (harness: typeof ordered[number]["id"]) => ordered.find(({ id }) => id === harness)!,
      }
      const service = yield* makeHarnessConnectionService({
        paths,
        registry,
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })
      yield* service.connect(HarnessIdSchema.make("pi"), {
        model: Option.some(model),
        installSkill: false,
      })
      const connectedSettings = yield* fs.readFileString(paths.piSettings)
      const connectedModels = yield* fs.readFileString(paths.piModels)

      const exit = yield* Effect.exit(service.disconnect(HarnessIdSchema.make("pi")))

      expect(exit._tag).toBe("Failure")
      expect(yield* fs.readFileString(paths.piSettings)).toBe(connectedSettings)
      expect(yield* fs.readFileString(paths.piModels)).toBe(connectedModels)
      expect(readJson(yield* fs.readFileString(paths.manifest))).toMatchObject({
        connections: [{ harness: "pi", companion: { ownership: "magnitude" } }],
      })
      expect((yield* fs.readFileString(`${root}/pi/package-commands.jsonl`)).trim().split("\n").map((line) => JSON.parse(line)))
        .toEqual([
          { command: "install", source: PI_COMPANION_PACKAGE_SOURCE },
          { command: "remove", source: PI_COMPANION_PACKAGE_SOURCE },
          { command: "install", source: PI_COMPANION_PACKAGE_SOURCE },
        ])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("leaves connection state unchanged when Pi rejects package installation", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-install-failure-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const executable = yield* writeFakePiExecutable(paths, "install")
      const service = yield* makeHarnessConnectionService({
        paths,
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })

      const exit = yield* Effect.exit(service.connect(HarnessIdSchema.make("pi"), {
        model: Option.some(model),
        installSkill: true,
      }))

      expect(exit._tag).toBe("Failure")
      expect(yield* fs.readFileString(paths.piSettings)).toBe(`${initial[paths.piSettings]}\n`)
      expect(yield* fs.readFileString(paths.piModels)).toBe(`${initial[paths.piModels]}\n`)
      expect(yield* fs.exists(paths.skillInstallations["shared-agents"].skillFile)).toBe(false)
      expect(readJson(yield* fs.readFileString(paths.manifest))).toEqual({ connections: [] })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("retains a connected package and manifest when Pi rejects package removal", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-pi-package-remove-failure-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const executable = yield* writeFakePiExecutable(paths, "remove")
      const service = yield* makeHarnessConnectionService({
        paths,
        detect: () => Effect.succeed(Option.some({ executable })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })
      yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.some(model) })
      const connectedSettings = yield* fs.readFileString(paths.piSettings)
      const connectedModels = yield* fs.readFileString(paths.piModels)

      const exit = yield* Effect.exit(service.disconnect(HarnessIdSchema.make("pi")))

      expect(exit._tag).toBe("Failure")
      expect(yield* fs.readFileString(paths.piSettings)).toBe(connectedSettings)
      expect(yield* fs.readFileString(paths.piModels)).toBe(connectedModels)
      expect(readJson(yield* fs.readFileString(paths.manifest))).toMatchObject({
        connections: [{ harness: "pi", companion: { ownership: "magnitude" } }],
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })
})

describe("Magnitude skill installation", () => {
  it("shares one installation and replaces it on every interoperable-harness install", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-shared-skill-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths)

      yield* service.installSkill(HarnessIdSchema.make("pi"))
      const sharedSkill = paths.skillInstallations["shared-agents"].skillFile
      const installed = yield* fs.readFileString(sharedSkill)
      expect(installed).toContain("name: magnitude")
      expect(installed).toContain("magnitude docs onboarding")
      yield* fs.writeFileString(sharedSkill, "stale contents\n")
      yield* service.installSkill(HarnessIdSchema.make("codex"))

      expect(yield* fs.readFileString(sharedSkill)).toBe(installed)
      expect(yield* fs.exists(paths.skillInstallations["hermes-user"].skillFile)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("uses specialized installations only for harnesses that require them", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-specialized-skill-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths)

      yield* service.installSkill(HarnessIdSchema.make("hermes"))

      expect(yield* fs.readFileString(paths.skillInstallations["hermes-user"].skillFile))
        .toContain("name: magnitude")
      expect(yield* fs.exists(paths.skillInstallations["shared-agents"].skillFile)).toBe(false)
      expect(yield* fs.exists(paths.skillInstallations["claude-user"].skillFile)).toBe(false)
      expect(yield* fs.exists(paths.skillInstallations["cline-user"].skillFile)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

})

describe("HarnessConnection model-set behavior", () => {
  it("reads fresh manifests across independent service instances", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-manifest-concurrent-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const first = yield* installedService(paths)
      const second = yield* installedService(paths)
      yield* Effect.all([
        first.connect(HarnessIdSchema.make("pi"), { model: Option.none() }),
        second.connect(HarnessIdSchema.make("opencode"), { model: Option.none() }),
      ], { concurrency: "unbounded" })
      expect((yield* first.list).filter((row) => row.connected).map((row) => row.id)).toEqual(["pi", "opencode"])
      yield* second.disconnect(HarnessIdSchema.make("pi"))
      expect((yield* first.list).filter((row) => row.connected).map((row) => row.id)).toEqual(["opencode"])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })
  it("preserves Hermes global and explicit per-model overrides", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-hermes-reasoning-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({
        ...initialFiles(paths),
        [paths.hermes]: `agent:\n  reasoning_effort: medium\n  reasoning_overrides:\n    "local/model": low\n`,
      })
      const service = yield* installedService(paths)
      yield* service.connect(HarnessIdSchema.make("hermes"), { model: Option.some(model) })

      expect(readYaml(yield* fs.readFileString(paths.hermes))).toMatchObject({
        agent: {
          reasoning_effort: "medium",
          reasoning_overrides: {
            [model]: "low",
            [secondModel]: "none",
          },
        },
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("ensures startup before Claude connect and sync only", async () => {
    let startupCalls = 0
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-startup-" })
      const paths = fixturePaths(root)
      const service = yield* makeHarnessConnectionService({
        paths,
        registry: makeHarnessConnectorRegistry(paths, { piCompanion: testPiCompanion }),
        detect: (connector) => Effect.succeed(Option.some({ executable: `/installed/${connector.id}` })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.sync(() => { startupCalls += 1 }),
      })

      yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.some(model) })
      expect(startupCalls).toBe(0)
      yield* service.connect(HarnessIdSchema.make("claude-code"), { model: Option.some(model) })
      expect(startupCalls).toBe(1)
      yield* service.sync(HarnessIdSchema.make("claude-code"))
      expect(startupCalls).toBe(2)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("reconciles repeated setup and selected-model changes for every harness", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-idempotent-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const service = yield* installedService(paths)
      const connectors = makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")
      const readConfiguration = (file: string) => fs.readFileString(file).pipe(
        Effect.map(Option.some),
        Effect.catchTag("SystemError", (error) => error.reason === "NotFound"
          ? Effect.succeed(Option.none<string>())
          : Effect.fail(error)),
      )

      for (const connector of connectors) {
        yield* service.connect(connector.id, { model: Option.some(model) })
        const first = yield* Effect.forEach(connector.configurationFiles, readConfiguration)
        yield* service.connect(connector.id, { model: Option.some(model) })
        const repeated = yield* Effect.forEach(connector.configurationFiles, readConfiguration)
        expect(repeated, connector.id).toEqual(first)
        yield* service.connect(connector.id, { model: Option.some(secondModel) })
      }

      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        defaultProvider: "magnitude", defaultModel: secondModel,
      })
      expect(readJson(yield* fs.readFileString(paths.opencode))).toMatchObject({ model: `magnitude/${secondModel}` })
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toMatchObject({
        model: { provider: "custom:magnitude", default: secondModel },
      })
      expect(readYaml(yield* fs.readFileString(paths.ompSettings))).toMatchObject({
        modelRoles: { default: `magnitude/${secondModel}` },
      })

      for (const connector of connectors) {
        yield* service.disconnect(connector.id)
        yield* service.disconnect(connector.id)
      }
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toEqual(readJson(initial[paths.piSettings]!))
      expect(readJson(yield* fs.readFileString(paths.opencode))).toEqual(readJson(initial[paths.opencode]!))
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toEqual(readYaml(initial[paths.hermes]!))
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toEqual(readJson(initial[paths.openclaw]!))
      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toEqual(Bun.TOML.parse(initial[paths.codexUser]!))
      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual(readJson(initial[paths.claude]!))
      expect(readYaml(yield* fs.readFileString(paths.ompSettings))).toEqual(readYaml(initial[paths.ompSettings]!))
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toEqual(readJson(initial[paths.clineProviders]!))
      expect((readJson(yield* fs.readFileString(paths.manifest)) as { connections: unknown[] }).connections).toEqual([])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("replaces arbitrary pre-existing Magnitude-owned entries", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-replace-owned-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures({
        ...initial,
        [paths.piModels]: stringifyJson({ providers: { magnitude: "old" } }),
        [paths.opencode]: stringifyJson({ provider: { magnitude: "old" }, model: "user/model" }),
        [paths.hermes]: "providers:\n  magnitude: old\n",
        [paths.openclaw]: stringifyJson({
          models: { providers: { magnitude: "old" } },
          agents: { list: [{ id: "magnitude", model: "old" }] },
        }),
        [paths.codex]: "unrecognized old contents\n",
        [paths.codexModels]: "unrecognized old contents\n",
        [paths.ompModels]: "providers:\n  magnitude: old\n",
        [paths.clineProviders]: stringifyJson({
          version: 1, modes: {}, providers: { "openai-compatible": { old: true } },
        }),
        [paths.clineModels]: stringifyJson({
          version: 1, providers: { "openai-compatible": { old: true } },
        }),
      })
      const service = yield* installedService(paths)

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        yield* service.connect(connector.id, { model: Option.some(model) })
      }

      expect(readJson(yield* fs.readFileString(paths.piModels))).toMatchObject({
        providers: { magnitude: piProviderConfig(models) },
      })
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toMatchObject({
        providers: { "openai-compatible": { settings: clineProviderSettings(Option.some(model)) } },
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("connects every model without changing current selections when a model is absent", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-all-models-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const service = yield* installedService(paths)

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        yield* service.connect(connector.id, { model: Option.none() })
      }

      expect(readJson(yield* fs.readFileString(paths.piModels))).toMatchObject({
        providers: { magnitude: piProviderConfig(models) },
      })
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toEqual(readJson(initial[paths.piSettings]!))
      expect(readJson(yield* fs.readFileString(paths.opencode))).toMatchObject({
        model: "user/model", provider: { magnitude: openCodeProviderConfig(models) },
      })
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toEqual({
        theme: "dark",
        model: { provider: "user", default: "model" },
        providers: { magnitude: hermesProviderConfig() },
        agent: { reasoning_overrides: hermesReasoningOverrides(models) },
      })
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toMatchObject({
        models: { providers: { magnitude: openClawProviderConfig(models) } },
        agents: { defaults: { model: { primary: "user/model" } }, list: [{ id: "main", model: "user/model" }] },
      })
      const codexSpec = {
        models,
        model: Option.none<typeof model>(),
        installation: { executable: "/installed/codex" },
      }
      expect(yield* fs.exists(paths.codex)).toBe(false)
      expect(yield* fs.readFileString(paths.codexModels)).toBe(
        codexModelCatalog(codexSpec, bundledCodexCatalog),
      )
      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toMatchObject({
        model_provider: "magnitude", model: "user/model",
        model_catalog_json: paths.codexModels,
        model_providers: {
          magnitude: {
            name: "OpenAI",
            base_url: CODEX_PROXY_BASE_URL,
            wire_api: "responses",
            requires_openai_auth: true,
            supports_websockets: true,
          },
        },
      })
      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual({
        theme: "dark",
        model: "user/model",
        env: {
          USER_KEY: "preserve",
          ANTHROPIC_BASE_URL,
          CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY: CLAUDE_GATEWAY_DISCOVERY,
        },
      })
      expect(readYaml(yield* fs.readFileString(paths.ompModels))).toMatchObject({
        providers: { magnitude: ohMyPiProviderConfig(models) },
      })
      expect(readYaml(yield* fs.readFileString(paths.ompSettings))).toEqual(readYaml(initial[paths.ompSettings]!))
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toMatchObject({
        lastUsedProvider: "user-provider",
        providers: { "openai-compatible": { settings: clineProviderSettings(Option.none()) } },
      })
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toMatchObject({
        providers: { "openai-compatible": clineModelRegistryEntry(models) },
      })

      const manifest = readJson(yield* fs.readFileString(paths.manifest)) as {
        connections: ReadonlyArray<Record<string, unknown>>
      }
      expect(manifest.connections).toHaveLength(8)
      for (const entry of manifest.connections) {
        expect(entry.models).toEqual(models)
        if (entry.harness === "codex") expect(entry).toHaveProperty("restore")
        else expect(entry).not.toHaveProperty("restore")
      }

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        yield* service.disconnect(connector.id)
      }
      expect(readJson(yield* fs.readFileString(paths.piModels))).toEqual(readJson(initial[paths.piModels]!))
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toEqual(readJson(initial[paths.piSettings]!))
      expect(readJson(yield* fs.readFileString(paths.opencode))).toEqual(readJson(initial[paths.opencode]!))
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toEqual(readYaml(initial[paths.hermes]!))
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toEqual(readJson(initial[paths.openclaw]!))
      expect(yield* fs.exists(paths.codex)).toBe(false)
      expect(yield* fs.exists(paths.codexModels)).toBe(false)
      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toEqual(Bun.TOML.parse(initial[paths.codexUser]!))
      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual(readJson(initial[paths.claude]!))
      expect(readYaml(yield* fs.readFileString(paths.ompModels))).toEqual(readYaml(initial[paths.ompModels]!))
      expect(readYaml(yield* fs.readFileString(paths.ompSettings))).toEqual(readYaml(initial[paths.ompSettings]!))
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toEqual(readJson(initial[paths.clineProviders]!))
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toEqual(readJson(initial[paths.clineModels]!))
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("observes durable connection intent independently of harness installation", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-observation-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths)
      const harness = HarnessIdSchema.make("pi")

      expect((yield* service.list).find(({ id }) => id === harness)?.connected).toBe(false)
      yield* service.connect(harness, { model: Option.none() })
      expect((yield* service.list).find(({ id }) => id === harness)?.connected).toBe(true)
      yield* service.disconnect(harness)
      yield* service.disconnect(harness)
      expect((yield* service.list).find(({ id }) => id === harness)?.connected).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("persists model selection, launches independently, and restores on disconnect", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-current-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const service = yield* installedService(paths)

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        yield* service.connect(connector.id, { model: Option.some(model) })
        const plan = yield* service.launch(connector.id, model)
        expect(plan.harness).toBe(connector.id)
        expect(plan.command).toBe(connector.executable)
        expect(plan.executable).toBe(`/installed/${connector.id}`)
        expect(plan.modelId).toBe(model)
      }

      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        defaultProvider: "magnitude", defaultModel: model,
      })
      expect(readJson(yield* fs.readFileString(paths.opencode))).toMatchObject({ model: `magnitude/${model}` })
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toMatchObject({
        model: { provider: "custom:magnitude", default: model },
      })
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toMatchObject({
        agents: { defaults: { model: { primary: `magnitude/${model}` } } },
      })
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toMatchObject({
        agents: { list: [
          { id: "main", model: "user/model" },
          { id: "magnitude", model: `magnitude/${model}`, thinkingDefault: "high" },
        ] },
      })
      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toMatchObject({
        model_provider: "magnitude",
        model: codexLocalModelId(model),
        model_catalog_json: paths.codexModels,
        model_providers: {
          magnitude: {
            name: "OpenAI",
            base_url: CODEX_PROXY_BASE_URL,
            supports_websockets: true,
          },
        },
      })
      expect(yield* fs.exists(paths.codex)).toBe(false)
      const codexModels = readJson(yield* fs.readFileString(paths.codexModels)) as {
        future_root_field: unknown
        models: ReadonlyArray<Record<string, unknown>>
      }
      expect(codexModels.future_root_field).toEqual({ preserved: true })
      expect(codexModels.models[0]).toEqual(bundledCodexCatalog.models[0])
      expect(codexModels.models[1]).toMatchObject({
        slug: codexLocalModelId(model),
        display_name: "Local Model (Q4)",
        context_window: 50_000,
        default_reasoning_level: "high",
        supported_reasoning_levels: [{ effort: "high" }],
        service_tiers: [],
      })
      expect(readJson(yield* fs.readFileString(paths.claude))).toMatchObject({
        model: anthropicLocalModelId(model),
      })
      expect(readYaml(yield* fs.readFileString(paths.ompSettings))).toMatchObject({
        modelRoles: { default: `magnitude/${model}` },
      })
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toMatchObject({
        lastUsedProvider: "openai-compatible",
        providers: { "openai-compatible": { settings: clineProviderSettings(Option.some(model)) } },
      })

      const manifest = readJson(yield* fs.readFileString(paths.manifest)) as {
        connections: ReadonlyArray<Record<string, unknown>>
      }
      expect(manifest.connections.every((entry) => Object.hasOwn(entry, "restore"))).toBe(true)
      expect(manifest.connections.every((entry) => {
        const restore = entry.restore
        return typeof restore === "object"
          && restore !== null
          && Object.keys(restore).every((key) => key === "model")
      })).toBe(true)

      const openClawPlan = yield* service.launch(HarnessIdSchema.make("openclaw"), model)
      expect(openClawPlan.args.slice(0, 3)).toEqual(["tui", "--local", "--session"])
      expect(openClawPlan.args[3]).toMatch(/^agent:magnitude:[0-9a-f-]{36}$/)
      const codexPlan = yield* service.launch(HarnessIdSchema.make("codex"), model)
      expect(codexPlan.args).toEqual(["--model", codexLocalModelId(model)])

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        yield* service.disconnect(connector.id)
      }

      expect(readJson(yield* fs.readFileString(paths.piModels))).toEqual(readJson(initial[paths.piModels]!))
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toEqual(readJson(initial[paths.piSettings]!))
      expect(readJson(yield* fs.readFileString(paths.opencode))).toEqual(readJson(initial[paths.opencode]!))
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toEqual(readYaml(initial[paths.hermes]!))
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toEqual(readJson(initial[paths.openclaw]!))
      expect(yield* fs.exists(paths.codex)).toBe(false)
      expect(yield* fs.exists(paths.codexModels)).toBe(false)
      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toEqual(Bun.TOML.parse(initial[paths.codexUser]!))
      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual(readJson(initial[paths.claude]!))
      expect(readYaml(yield* fs.readFileString(paths.ompModels))).toEqual(readYaml(initial[paths.ompModels]!))
      expect(readYaml(yield* fs.readFileString(paths.ompSettings))).toEqual(readYaml(initial[paths.ompSettings]!))
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toEqual(readJson(initial[paths.clineProviders]!))
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toEqual(readJson(initial[paths.clineModels]!))
      expect((readJson(yield* fs.readFileString(paths.manifest)) as { connections: unknown[] }).connections).toEqual([])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("restores only the prior Codex model selection and clears the owned catalog", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-codex-restore-" })
      const paths = fixturePaths(root)
      const original = [
        'model_provider = "custom"',
        'model = "user/model"',
        'openai_base_url = "https://user.example/v1"',
        'model_catalog_json = "/user/catalog.json"',
        "",
      ].join("\n")
      yield* writeFixtures({ [paths.codexUser]: original })
      const service = yield* installedService(paths)

      yield* service.connect(HarnessIdSchema.make("codex"), { model: Option.some(model) })
      yield* service.disconnect(HarnessIdSchema.make("codex"))

      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toEqual({
        model_provider: "custom",
        model: "user/model",
        openai_base_url: "https://user.example/v1",
      })
      expect(yield* fs.exists(paths.codexModels)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("preserves user-edited Codex endpoint and catalog while still disconnecting", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-codex-diverged-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({
        ...initialFiles(paths),
        [paths.codexUser]: [
          'model_provider = "openai"',
          'model = "user/model"',
          `openai_base_url = "${CODEX_PROXY_BASE_URL}"`,
          "",
        ].join("\n"),
      })
      const service = yield* installedService(paths)

      yield* service.connect(HarnessIdSchema.make("codex"), { model: Option.some(model) })
      const connected = yield* fs.readFileString(paths.codexUser)
      yield* fs.writeFileString(paths.codexUser, connected
        .replace(`openai_base_url = "${CODEX_PROXY_BASE_URL}"`, 'openai_base_url = "https://new.example/v1"')
        .replace(`model_catalog_json = "${paths.codexModels}"`, 'model_catalog_json = "/new/catalog.json"'))
      yield* service.disconnect(HarnessIdSchema.make("codex"))

      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toMatchObject({
        model_provider: "openai",
        model: "user/model",
        openai_base_url: "https://new.example/v1",
        model_catalog_json: "/new/catalog.json",
      })
      expect(yield* fs.exists(paths.codexModels)).toBe(false)
      expect((readJson(yield* fs.readFileString(paths.manifest)) as { connections: unknown[] }).connections)
        .toEqual([])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("refreshes the installed Codex catalog on sync without changing selection or restoration", async () => {
    let bundled: Readonly<Record<string, unknown>> & { readonly models: ReadonlyArray<unknown> } =
      bundledCodexCatalog
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-codex-sync-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const registry = makeHarnessConnectorRegistry(paths, {
        readCodexBundledCatalog: () => Effect.sync(() => bundled),
      })
      const service = yield* makeHarnessConnectionService({
        paths,
        registry,
        detect: (connector) => Effect.succeed(Option.some({ executable: `/installed/${connector.id}` })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })

      yield* service.connect(HarnessIdSchema.make("codex"), { model: Option.some(model) })
      const restoreBefore = (readJson(yield* fs.readFileString(paths.manifest)) as {
        connections: ReadonlyArray<{ restore?: unknown }>
      }).connections[0]?.restore
      bundled = {
        future_root_field: { refreshed: true },
        models: [{ slug: "new-openai-fixture", unknown: "preserved" }],
      }
      yield* service.sync(HarnessIdSchema.make("codex"))

      const config = Bun.TOML.parse(yield* fs.readFileString(paths.codexUser)) as Record<string, unknown>
      expect(config.model).toBe(codexLocalModelId(model))
      expect(config.model_provider).toBe("magnitude")
      const catalog = readJson(yield* fs.readFileString(paths.codexModels)) as {
        future_root_field: unknown
        models: ReadonlyArray<Record<string, unknown>>
      }
      expect(catalog.future_root_field).toEqual({ refreshed: true })
      expect(catalog.models[0]).toEqual({ slug: "new-openai-fixture", unknown: "preserved" })
      const restoreAfter = (readJson(yield* fs.readFileString(paths.manifest)) as {
        connections: ReadonlyArray<{ restore?: unknown }>
      }).connections[0]?.restore
      expect(restoreAfter).toEqual(restoreBefore)

      yield* service.disconnect(HarnessIdSchema.make("codex"))
      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser)))
        .toEqual(Bun.TOML.parse(initial[paths.codexUser]!))
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("leaves Codex files unchanged when bundled catalog export fails", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-codex-export-failure-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures({
        ...initial,
        [paths.codex]: "legacy config\n",
        [paths.codexModels]: "legacy catalog\n",
      })
      const registry = makeHarnessConnectorRegistry(paths, {
        readCodexBundledCatalog: () => Effect.fail("export failed"),
      })
      const service = yield* makeHarnessConnectionService({
        paths,
        registry,
        detect: (connector) => Effect.succeed(Option.some({ executable: `/installed/${connector.id}` })),
        resolveModels: Effect.succeed(models),
        installStartup: Effect.void,
      })

      const result = yield* Effect.either(
        service.connect(HarnessIdSchema.make("codex"), { model: Option.some(model) }),
      )
      expect(result._tag).toBe("Left")
      expect(yield* fs.readFileString(paths.codexUser)).toBe(initial[paths.codexUser])
      expect(yield* fs.readFileString(paths.codex)).toBe("legacy config\n")
      expect(yield* fs.readFileString(paths.codexModels)).toBe("legacy catalog\n")
      expect((readJson(yield* fs.readFileString(paths.manifest)) as { connections: unknown[] }).connections)
        .toEqual([])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("removes Magnitude-owned entries even after they diverge", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-divergent-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths)
      yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.none() })
      const connected = readJson(yield* fs.readFileString(paths.piModels)) as Record<string, unknown>
      ;(connected.providers as Record<string, unknown>).magnitude = { userEdited: true }
      yield* fs.writeFileString(paths.piModels, stringifyJson(connected))
      yield* service.disconnect(HarnessIdSchema.make("pi"))
      expect(readJson(yield* fs.readFileString(paths.piModels))).toEqual({ providers: {} })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("skips model restoration after a user selects a non-Magnitude model and still disconnects", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-user-selection-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const service = yield* installedService(paths)
      const connectors = makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")
      for (const connector of connectors) yield* service.connect(connector.id, { model: Option.some(model) })

      const pi = yield* fs.readFileString(paths.piSettings)
      yield* fs.writeFileString(paths.piSettings, updateJsonc(pi, [
        [["defaultProvider"], "user"], [["defaultModel"], "new/model"],
      ]))
      const opencode = yield* fs.readFileString(paths.opencode)
      yield* fs.writeFileString(paths.opencode, updateJsonc(opencode, [[["model"], "user/new-model"]]))
      const hermes = yield* fs.readFileString(paths.hermes)
      yield* fs.writeFileString(paths.hermes, updateYaml(hermes, [
        [["model", "provider"], "user"], [["model", "default"], "new-model"],
      ]))
      const openclaw = yield* fs.readFileString(paths.openclaw)
      yield* fs.writeFileString(paths.openclaw, updateJsonc(openclaw, [[
        ["agents", "defaults", "model", "primary"], "user/new-model",
      ]]))
      const codex = (yield* fs.readFileString(paths.codexUser))
        .replace(`model = "${codexLocalModelId(model)}"`, 'model = "user/new-model"')
      yield* fs.writeFileString(paths.codexUser, codex)
      const claude = yield* fs.readFileString(paths.claude)
      yield* fs.writeFileString(paths.claude, updateJsonc(claude, [[["model"], "user/new-model"]]))
      const omp = yield* fs.readFileString(paths.ompSettings)
      yield* fs.writeFileString(paths.ompSettings, updateYaml(omp, [[
        ["modelRoles", "default"], "user/new-model",
      ]]))
      const cline = yield* fs.readFileString(paths.clineProviders)
      yield* fs.writeFileString(paths.clineProviders, updateJsonc(cline, [[
        ["lastUsedProvider"], "user-provider",
      ]]))

      yield* service.sync()
      for (const connector of connectors) yield* service.disconnect(connector.id)

      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        defaultProvider: "user", defaultModel: "new/model",
      })
      expect(readJson(yield* fs.readFileString(paths.opencode))).toMatchObject({ model: "user/new-model" })
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toMatchObject({
        model: { provider: "user", default: "new-model" },
      })
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toMatchObject({
        agents: { defaults: { model: { primary: "user/new-model" } } },
      })
      expect(Bun.TOML.parse(yield* fs.readFileString(paths.codexUser))).toMatchObject({
        model_provider: "openai", model: "user/new-model",
      })
      expect(readJson(yield* fs.readFileString(paths.claude))).toMatchObject({ model: "user/new-model" })
      expect(readYaml(yield* fs.readFileString(paths.ompSettings))).toMatchObject({
        modelRoles: { default: "user/new-model" },
      })
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toMatchObject({
        lastUsedProvider: "user-provider",
      })
      expect((readJson(yield* fs.readFileString(paths.manifest)) as { connections: unknown[] }).connections).toEqual([])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("recovers invalid manifest entries without blocking setup", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-recovery-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({
        [paths.manifest]: stringifyJson({
          version: 2,
          connections: [
            {
              harness: "pi",
              models: [model],
              setModel: model,
              updatedAt: "2026-08-25T00:00:00.000Z",
            },
          ],
        }),
      })

      const service = yield* installedService(paths)
      expect((yield* service.list).length).toBe(HARNESS_PRIORITY.length)
      expect(readJson(yield* fs.readFileString(paths.manifest))).toEqual({
        connections: [{
          harness: "pi",
          models: [],
          updatedAt: "2026-08-25T00:00:00.000Z",
        }],
      })

      yield* service.connect(HarnessIdSchema.make("pi"), { model: Option.some(model) })
      expect(readJson(yield* fs.readFileString(paths.manifest))).toMatchObject({
        connections: [{ harness: "pi", models, restore: { } }],
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("recovers manifest roots, entries, and fields at their nearest valid boundary", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-recursive-recovery-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({
        [paths.manifest]: stringifyJson({
          connections: [
            {
              harness: "pi",
              models: [models[0], { ...models[1], name: 42 }],
              restore: { model: 42 },
              updatedAt: 42,
            },
            { harness: "unknown", models, updatedAt: "2026-08-25T00:00:00.000Z" },
            {
              harness: "codex",
              models,
              updatedAt: "2026-08-25T00:00:00.000Z",
            },
          ],
        }),
      })

      yield* installedService(paths)

      expect(readJson(yield* fs.readFileString(paths.manifest))).toEqual({
        connections: [
          {
            harness: "pi",
            models: [models[0]],
            restore: {},
            updatedAt: "1970-01-01T00:00:00.000Z",
          },
          {
            harness: "codex",
            models,
            updatedAt: "2026-08-25T00:00:00.000Z",
          },
        ],
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("preserves malformed and invalid-root manifests before installing the root default", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-root-recovery-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({ [paths.manifest]: "{" })

      yield* installedService(paths)

      expect(readJson(yield* fs.readFileString(paths.manifest))).toEqual({ connections: [] })
      expect((yield* fs.readDirectory(paths.manifest.slice(0, paths.manifest.lastIndexOf("/"))))
        .filter((name) => name.startsWith("harness-connections.json.corrupt-"))).toHaveLength(1)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))

    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-invalid-root-" })
      const paths = fixturePaths(root)
      yield* writeFixtures({
        [paths.manifest]: stringifyJson({ connections: "wrong", unrelated: true }),
      })

      yield* installedService(paths)

      expect(readJson(yield* fs.readFileString(paths.manifest))).toEqual({ connections: [] })
      expect((yield* fs.readDirectory(paths.manifest.slice(0, paths.manifest.lastIndexOf("/"))))
        .filter((name) => name.startsWith("harness-connections.json.corrupt-"))).toHaveLength(1)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("removes Cline's isolated provider and model catalog even after edits", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-cline-divergent-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths)
      yield* service.connect(HarnessIdSchema.make("cline"), { model: Option.none() })

      const providers = readJson(yield* fs.readFileString(paths.clineProviders)) as {
        providers: Record<string, unknown>
      }
      providers.providers["openai-compatible"] = { userEdited: true }
      yield* fs.writeFileString(paths.clineProviders, stringifyJson(providers))

      yield* service.disconnect(HarnessIdSchema.make("cline"))

      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toEqual({
        version: 1, modes: {}, providers: {},
      })
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toEqual({ version: 1, providers: {} })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("preserves later Claude gateway edits during disconnect", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-claude-edit-" })
      const paths = fixturePaths(root)
      yield* writeFixtures(initialFiles(paths))
      const service = yield* installedService(paths)
      yield* service.connect(HarnessIdSchema.make("claude-code"), { model: Option.some(model) })
      const settings = readJson(yield* fs.readFileString(paths.claude)) as { env: Record<string, string> }
      settings.env.ANTHROPIC_BASE_URL = "https://user-gateway.example"
      yield* fs.writeFileString(paths.claude, stringifyJson(settings))

      yield* service.disconnect(HarnessIdSchema.make("claude-code"))

      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual({
        theme: "dark",
        model: anthropicLocalModelId(model),
        env: { USER_KEY: "preserve", ANTHROPIC_BASE_URL: "https://user-gateway.example" },
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("sync refreshes Cline's custom Magnitude catalog", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-cline-sync-" })
      const paths = fixturePaths(root)
      const first = yield* installedService(paths, [models[0]])
      yield* first.connect(HarnessIdSchema.make("cline"), { model: Option.some(model) })
      const second = yield* installedService(paths, models)
      yield* second.sync(HarnessIdSchema.make("cline"))

      expect(readJson(yield* fs.readFileString(paths.clineModels))).toMatchObject({
        providers: { "openai-compatible": clineModelRegistryEntry(models) },
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("sync refreshes the complete model set without changing model selection", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-sync-" })
      const paths = fixturePaths(root)
      const first = yield* installedService(paths, [models[0]])
      yield* first.connect(HarnessIdSchema.make("pi"), { model: Option.some(model) })
      const second = yield* installedService(paths, models)
      yield* second.sync(HarnessIdSchema.make("pi"))
      expect(readJson(yield* fs.readFileString(paths.piModels))).toMatchObject({
        providers: { magnitude: piProviderConfig(models) },
      })
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        defaultProvider: "magnitude", defaultModel: model,
      })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("rejects a model when it is not in the installed Magnitude model set", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-invalid-current-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths, [models[1]])
      const exit = yield* Effect.exit(service.connect(
        HarnessIdSchema.make("pi"),
        { model: Option.some(model) },
      ))
      expect(exit._tag).toBe("Failure")
      expect(readJson(yield* fs.readFileString(paths.manifest))).toEqual({ connections: [] })
      expect(yield* fs.exists(paths.piModels)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("uses a launch-scoped model choice with persistent Claude gateway settings", () => {
    const connector = makeHarnessConnectorRegistry(fixturePaths("/tmp/claude")).get(HarnessIdSchema.make("claude-code"))
    const plan = connector.launch(model, { executable: "claude" })
    expect(plan.args).toEqual(["--model", anthropicLocalModelId(model)])
    expect(plan.environment).toEqual({})
  })

  it("launches Cline with an explicit model independently of persistent configuration", () => {
    const paths = fixturePaths("/tmp/cline")
    const connector = makeHarnessConnectorRegistry(paths).get(HarnessIdSchema.make("cline"))
    const plan = connector.launch(model, { executable: "cline" })
    expect(plan.args).toEqual([
      "--tui",
      "--provider",
      "openai-compatible",
      "--model",
      model,
    ])
  })
})
