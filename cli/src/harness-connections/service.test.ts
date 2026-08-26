import { FetchHttpClient } from "@effect/platform"
import * as FileSystem from "@effect/platform/FileSystem"
import { BunContext } from "@effect/platform-bun"
import { HARNESS_PRIORITY, HarnessIdSchema } from "@magnitudedev/client-common"
import { ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/sdk"
import { Effect, Option, Schema } from "effect"
import { parse } from "jsonc-parser"
import { delimiter } from "node:path"
import { parseDocument } from "yaml"
import { describe, expect, it } from "vitest"
import {
  ANTHROPIC_BASE_URL,
  OPENAI_BASE_URL,
  anthropicLocalModelId,
  clineModelCatalog,
  clineProviderSettings,
  codexConfig,
  codexModelCatalog,
  harnessExecutableSearchPath,
  hermesProviderConfig,
  makeHarnessConnectionService,
  makeHarnessConnectorRegistry,
  ohMyPiProviderConfig,
  openClawAgentConfig,
  openClawProviderConfig,
  openCodeProviderConfig,
  piProviderConfig,
  updateJsonc,
  type HarnessConnectionPaths,
} from "./service"

const model = ProviderModelIdSchema.make("local/model")
const secondModel = ProviderModelIdSchema.make("local/second-model")
const high = ReasoningEffortSchema.make("high")
const models = [
  {
    id: model,
    name: "Local Model (Q4)",
    description: "A local test model.",
    contextWindow: 50_000,
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
    capabilities: {
      vision: true,
      tools: true,
      structuredOutput: true,
      reasoning: { supported: false, efforts: [] },
    },
  },
] as const

const fixturePaths = (root: string): HarnessConnectionPaths => ({
  manifest: `${root}/magnitude/harness-connections.json`,
  piModels: `${root}/pi/models.json`,
  piSettings: `${root}/pi/settings.json`,
  opencode: `${root}/opencode/opencode.json`,
  hermes: `${root}/hermes/config.yaml`,
  openclaw: `${root}/openclaw/openclaw.json`,
  codex: `${root}/codex/magnitude.config.toml`,
  codexModels: `${root}/codex/magnitude.models.json`,
  claude: `${root}/claude/settings.json`,
  ompModels: `${root}/omp/models.yml`,
  ompSettings: `${root}/omp/settings.json`,
  clineProviders: `${root}/cline/providers.json`,
  clineModels: `${root}/cline/models.json`,
  skills: Object.fromEntries(HARNESS_PRIORITY.map((id) => [id, `${root}/skills/${id}/SKILL.md`])),
})

const stringifyJson = Schema.encodeSync(Schema.parseJson(Schema.Unknown, { space: 2 }))
const readJson = (source: string): unknown => parse(source)
const readYaml = (source: string): unknown => parseDocument(source).toJS()

const installedService = (paths: HarnessConnectionPaths, resolvedModels = models as ReadonlyArray<(typeof models)[number]>) =>
  makeHarnessConnectionService({
    paths,
    detect: (connector) => Effect.succeed(Option.some({ executable: `/installed/${connector.id}` })),
    resolveModels: Effect.succeed(resolvedModels),
    now: () => new Date("2026-08-26T00:00:00.000Z"),
  })

const initialFiles = (paths: HarnessConnectionPaths): Readonly<Record<string, string>> => ({
  [paths.piModels]: stringifyJson({ theme: "dark", providers: {} }),
  [paths.piSettings]: stringifyJson({ theme: "dark", defaultProvider: "user", defaultModel: "user/model" }),
  [paths.opencode]: stringifyJson({ theme: "dark", provider: {}, model: "user/model" }),
  [paths.hermes]: "theme: dark\n",
  [paths.openclaw]: stringifyJson({
    theme: "dark",
    models: { providers: {} },
    agents: {
      defaults: { model: { primary: "user/model" } },
      list: [{ id: "main", model: "user/model" }],
    },
  }),
  [paths.claude]: stringifyJson({ theme: "dark", env: { USER_KEY: "preserve" } }),
  [paths.ompModels]: "theme: dark\n",
  [paths.ompSettings]: stringifyJson({ theme: "dark", model: "user/model" }),
  [paths.clineProviders]: stringifyJson({
    version: 1,
    modes: { keep: true },
    providers: {},
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
      expect(typeof connector.installSkill).toBe("function")
    }
  })

  it("does not detect repository dependencies as installed harnesses", () => {
    expect(harnessExecutableSearchPath([
      "/workspace/node_modules/.bin", "/Users/developer/node_modules/.bin",
      "/Users/developer/.local/bin", "/usr/bin",
    ].join(delimiter))).toBe(["/Users/developer/.local/bin", "/usr/bin"].join(delimiter))
  })

  it("uses all models in model-enumerating provider formats", () => {
    expect(piProviderConfig(models).models).toEqual(models)
    expect(openCodeProviderConfig(models).models).toEqual({
      [model]: { name: "Local Model (Q4)" },
      [secondModel]: { name: "Second Model (Q6)" },
    })
    expect(openClawProviderConfig(models).models).toEqual(models)
    expect(ohMyPiProviderConfig(models).models).toEqual(models)
    expect(clineModelCatalog(models)).toEqual({
      [model]: { id: model, name: "Local Model (Q4)" },
      [secondModel]: { id: secondModel, name: "Second Model (Q6)" },
    })
    expect(hermesProviderConfig().transport).toBe("chat_completions")
    expect(clineProviderSettings(Option.none())).not.toHaveProperty("model")
    expect(clineProviderSettings(Option.some(model))).toHaveProperty("model", model)
  })

  it("keeps protocol assignments explicit", () => {
    expect(OPENAI_BASE_URL).toBe("http://127.0.0.1:10100/inference/v1")
    expect(ANTHROPIC_BASE_URL).toBe("http://127.0.0.1:10100/inference/anthropic")
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
})

describe("HarnessConnection model-set behavior", () => {
  it("connects every model without changing current selections when setCurrent is absent", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-all-models-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const service = yield* installedService(paths)

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        const result = yield* service.connect(connector.id, { setCurrent: Option.none() })
        expect(Option.isNone(result.launchPlan)).toBe(true)
      }

      expect(readJson(yield* fs.readFileString(paths.piModels))).toMatchObject({
        providers: { magnitude: piProviderConfig(models) },
      })
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toEqual(readJson(initial[paths.piSettings]!))
      expect(readJson(yield* fs.readFileString(paths.opencode))).toMatchObject({
        model: "user/model", provider: { magnitude: openCodeProviderConfig(models) },
      })
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toEqual({
        theme: "dark", providers: { magnitude: hermesProviderConfig() },
      })
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toMatchObject({
        models: { providers: { magnitude: openClawProviderConfig(models) } },
        agents: { defaults: { model: { primary: "user/model" } }, list: [{ id: "main", model: "user/model" }] },
      })
      const codexSpec = { models, setCurrent: Option.none<typeof model>() }
      expect(yield* fs.readFileString(paths.codex)).toBe(codexConfig(codexSpec, paths.codexModels))
      expect(yield* fs.readFileString(paths.codexModels)).toBe(codexModelCatalog(codexSpec))
      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual(readJson(initial[paths.claude]!))
      expect(readYaml(yield* fs.readFileString(paths.ompModels))).toMatchObject({
        providers: { magnitude: ohMyPiProviderConfig(models) },
      })
      expect(readJson(yield* fs.readFileString(paths.ompSettings))).toEqual(readJson(initial[paths.ompSettings]!))
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toMatchObject({
        lastUsedProvider: "user-provider",
        providers: { "openai-compatible": { settings: clineProviderSettings(Option.none()) } },
      })
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toMatchObject({
        providers: { "openai-compatible": { models: clineModelCatalog(models) } },
      })

      const manifest = readJson(yield* fs.readFileString(paths.manifest)) as {
        connections: ReadonlyArray<Record<string, unknown>>
      }
      expect(manifest.connections).toHaveLength(8)
      for (const entry of manifest.connections) {
        expect(entry.models).toEqual(models)
        expect(entry).not.toHaveProperty("setCurrent")
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
      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual(readJson(initial[paths.claude]!))
      expect(readYaml(yield* fs.readFileString(paths.ompModels))).toEqual(readYaml(initial[paths.ompModels]!))
      expect(readJson(yield* fs.readFileString(paths.ompSettings))).toEqual(readJson(initial[paths.ompSettings]!))
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toEqual(readJson(initial[paths.clineProviders]!))
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toEqual(readJson(initial[paths.clineModels]!))
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("applies setCurrent, returns exact launch plans, and disconnects every harness cleanly", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-current-" })
      const paths = fixturePaths(root)
      const initial = initialFiles(paths)
      yield* writeFixtures(initial)
      const service = yield* installedService(paths)

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        const result = yield* service.connect(connector.id, { setCurrent: Option.some(model) })
        expect(Option.isSome(result.launchPlan)).toBe(true)
        if (Option.isSome(result.launchPlan)) {
          expect(result.launchPlan.value.harness).toBe(connector.id)
          expect(result.launchPlan.value.modelId).toBe(model)
        }
      }

      expect(readJson(yield* fs.readFileString(paths.piSettings))).toMatchObject({
        defaultProvider: "magnitude", defaultModel: model,
      })
      expect(readJson(yield* fs.readFileString(paths.opencode))).toMatchObject({ model: `magnitude/${model}` })
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toMatchObject({
        model: { default: model, provider: "custom:magnitude" },
      })
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toMatchObject({
        agents: { list: [{ id: "main", model: "user/model" }, openClawAgentConfig(model)] },
      })
      expect(yield* fs.readFileString(paths.codex)).toContain(`model = "${model}"`)
      expect(yield* fs.readFileString(paths.codex)).toContain('model_reasoning_effort = "high"')
      expect(yield* fs.readFileString(paths.codex)).toContain('service_tier = "default"')
      const codexModels = readJson(yield* fs.readFileString(paths.codexModels)) as {
        models: ReadonlyArray<Record<string, unknown>>
      }
      expect(codexModels.models[0]).toMatchObject({
        slug: model,
        display_name: "Local Model (Q4)",
        context_window: 50_000,
        default_reasoning_level: "high",
        supported_reasoning_levels: [{ effort: "high" }],
        service_tiers: [],
      })
      expect(readJson(yield* fs.readFileString(paths.ompSettings))).toMatchObject({ model: `magnitude/${model}` })
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toMatchObject({
        providers: { "openai-compatible": { settings: clineProviderSettings(Option.some(model)) } },
      })

      for (const connector of makeHarnessConnectorRegistry(paths).ordered.filter(({ id }) => id !== "magnitude")) {
        yield* service.disconnect(connector.id)
      }

      expect(readJson(yield* fs.readFileString(paths.piModels))).toEqual(readJson(initial[paths.piModels]!))
      expect(readJson(yield* fs.readFileString(paths.piSettings))).toEqual({ theme: "dark" })
      expect(readJson(yield* fs.readFileString(paths.opencode))).toEqual({ theme: "dark", provider: {} })
      expect(readYaml(yield* fs.readFileString(paths.hermes))).toEqual(readYaml(initial[paths.hermes]!))
      expect(readJson(yield* fs.readFileString(paths.openclaw))).toEqual(readJson(initial[paths.openclaw]!))
      expect(yield* fs.exists(paths.codex)).toBe(false)
      expect(yield* fs.exists(paths.codexModels)).toBe(false)
      expect(readJson(yield* fs.readFileString(paths.claude))).toEqual(readJson(initial[paths.claude]!))
      expect(readYaml(yield* fs.readFileString(paths.ompModels))).toEqual(readYaml(initial[paths.ompModels]!))
      expect(readJson(yield* fs.readFileString(paths.ompSettings))).toEqual({ theme: "dark" })
      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toEqual(readJson(initial[paths.clineProviders]!))
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toEqual(readJson(initial[paths.clineModels]!))
      expect((readJson(yield* fs.readFileString(paths.manifest)) as { connections: unknown[] }).connections).toEqual([])
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("preserves divergent user edits during disconnect", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-divergent-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths)
      yield* service.connect(HarnessIdSchema.make("pi"), { setCurrent: Option.none() })
      const connected = readJson(yield* fs.readFileString(paths.piModels)) as Record<string, unknown>
      ;(connected.providers as Record<string, unknown>).magnitude = { userEdited: true }
      yield* fs.writeFileString(paths.piModels, stringifyJson(connected))
      yield* service.disconnect(HarnessIdSchema.make("pi"))
      expect(readJson(yield* fs.readFileString(paths.piModels))).toMatchObject({
        providers: { magnitude: { userEdited: true } },
      })
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
              setCurrent: model,
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
          setCurrent: model,
          updatedAt: "2026-08-25T00:00:00.000Z",
        }],
      })

      yield* service.connect(HarnessIdSchema.make("pi"), { setCurrent: Option.some(model) })
      expect(readJson(yield* fs.readFileString(paths.manifest))).toMatchObject({
        connections: [{ harness: "pi", models, setCurrent: model }],
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
              setCurrent: 42,
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

  it("removes Cline's owned model catalog even when its provider entry was edited", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-cline-divergent-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths)
      yield* service.connect(HarnessIdSchema.make("cline"), { setCurrent: Option.none() })

      const providers = readJson(yield* fs.readFileString(paths.clineProviders)) as {
        providers: Record<string, unknown>
      }
      providers.providers["openai-compatible"] = { userEdited: true }
      yield* fs.writeFileString(paths.clineProviders, stringifyJson(providers))

      yield* service.disconnect(HarnessIdSchema.make("cline"))

      expect(readJson(yield* fs.readFileString(paths.clineProviders))).toMatchObject({
        providers: { "openai-compatible": { userEdited: true } },
      })
      expect(readJson(yield* fs.readFileString(paths.clineModels))).toEqual({ version: 1, providers: {} })
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("sync refreshes the complete model set without changing setCurrent", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-sync-" })
      const paths = fixturePaths(root)
      const first = yield* installedService(paths, [models[0]])
      yield* first.connect(HarnessIdSchema.make("pi"), { setCurrent: Option.some(model) })
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

  it("rejects setCurrent when it is not in the installed Magnitude model set", async () => {
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-harness-invalid-current-" })
      const paths = fixturePaths(root)
      const service = yield* installedService(paths, [models[1]])
      const exit = yield* Effect.exit(service.connect(
        HarnessIdSchema.make("pi"),
        { setCurrent: Option.some(model) },
      ))
      expect(exit._tag).toBe("Failure")
      expect(readJson(yield* fs.readFileString(paths.manifest))).toEqual({ connections: [] })
      expect(yield* fs.exists(paths.piModels)).toBe(false)
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]))))
  })

  it("uses launch-scoped model aliases for Claude Code", () => {
    const connector = makeHarnessConnectorRegistry(fixturePaths("/tmp/claude")).get(HarnessIdSchema.make("claude-code"))
    const plan = connector.launch(model, { executable: "claude" })
    expect(plan.args).toEqual(["--model", anthropicLocalModelId(model)])
    expect(plan.environment).toMatchObject({
      ANTHROPIC_MODEL: anthropicLocalModelId(model),
      ANTHROPIC_DEFAULT_HAIKU_MODEL: anthropicLocalModelId(model),
      ANTHROPIC_DEFAULT_SONNET_MODEL: anthropicLocalModelId(model),
      ANTHROPIC_DEFAULT_OPUS_MODEL: anthropicLocalModelId(model),
    })
  })
})
