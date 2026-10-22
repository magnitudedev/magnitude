import { BunFileSystem, BunPath } from "@effect/platform-bun"
import { Effect, Layer, Option, Schema } from "effect"
import { afterEach, describe, expect, it } from "vitest"
import { mkdtempSync, rmSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import {
  CustomModelsConfigSchema,
  GlobalStorage,
  ProjectStorage,
  StorageLive,
  VersionLive,
  makeGlobalStorage,
  makeProjectStorage,
  type CustomModelsConfig,
} from "@magnitudedev/storage"
import { CustomFormModelIdSchema } from "@magnitudedev/acn-protocol"
import {
  configuredDrafterChoice,
  CustomModels,
  CustomModelsLive,
  isHidden,
  localModelsConfiguration,
  withCanonicalReferences,
  withDrafterChoice,
  withLocationEnabled,
} from "./custom-models"

const local = "local:my-finetune-q4_k_m:9f2c1a7b3e04"

const sampleConfig = (): CustomModelsConfig => Schema.decodeUnknownSync(CustomModelsConfigSchema)({
  locations: { ollama: false },
  paths: ["/Volumes/models", "/Users/x/finetune.gguf"],
  added: { "hf:owner/repo-GGUF/model-Q4_K_M.gguf": { revision: "c0ffee", projector: "mmproj.gguf" } },
  hidden: ["hf:owner/hidden/model.gguf"],
  drafters: {
    "hf:owner/repo-GGUF/model-Q4_K_M.gguf": "hf:z-lab/Model-DFlash/dflash.gguf@abc",
    [local]: "none",
    "hf:owner/other/model.gguf": "/drafters/unknown.gguf",
    "hf:owner/third/model.gguf": "builtin",
  },
})

describe("local-models configuration snapshot", () => {
  it("converts the choices users write into roots, added models, and drafters chosen by reference", () => {
    const configuration = localModelsConfiguration(
      sampleConfig(),
      [
        { kind: "HuggingFaceCache", paths: [{ path: "/hf/hub", state: "Found" }], state: "Found" },
        { kind: "Ollama", paths: [
          { path: "/home/ada/.ollama/models", state: "Found" },
          { path: "/var/lib/ollama", state: "Unreadable" },
          { path: "/usr/share/ollama/.ollama/models", state: "NotFound" },
        ], state: "Found" },
      ],
    )
    expect(configuration.roots).toEqual([
      { kind: "HuggingFaceCache", path: "/hf/hub", enabled: true },
      { kind: "Ollama", path: "/home/ada/.ollama/models", enabled: false },
      { kind: "Ollama", path: "/var/lib/ollama", enabled: false },
      { kind: "Ollama", path: "/usr/share/ollama/.ollama/models", enabled: false },
      { kind: "Folder", path: "/Volumes/models", enabled: true },
      { kind: "Folder", path: "/Users/x/finetune.gguf", enabled: true },
    ])
    expect(configuration.added).toEqual([{
      repository: "owner/repo-GGUF",
      path: "model-Q4_K_M.gguf",
      revision: "c0ffee",
      projector: Option.some("mmproj.gguf"),
    }])
    expect(configuration.drafters).toEqual([
      {
        modelId: "hf:owner/repo-GGUF/model-Q4_K_M.gguf",
        drafter: { _tag: "Reference", reference: { _tag: "HuggingFace", repository: "z-lab/Model-DFlash", commit: "abc", path: "dflash.gguf" } },
      },
      { modelId: local, drafter: { _tag: "None" } },
      { modelId: "hf:owner/other/model.gguf", drafter: { _tag: "Reference", reference: { _tag: "Local", path: "/drafters/unknown.gguf" } } },
      { modelId: "hf:owner/third/model.gguf", drafter: { _tag: "Reference", reference: { _tag: "Embedded" } } },
    ])
  })

  it("finds a model's choices under its ID with any local: label, and stores automatic by leaving it out", () => {
    const config = sampleConfig()
    const relabeled = CustomFormModelIdSchema.make("local:my-finetune-v2-q4_k_m:9f2c1a7b3e04")
    expect(configuredDrafterChoice(config, relabeled)).toEqual({ _tag: "None" })
    expect(configuredDrafterChoice(config, CustomFormModelIdSchema.make("hf:owner/unknown/model.gguf"))).toEqual({ _tag: "Auto" })
    const automatic = withDrafterChoice(config, relabeled, { _tag: "Auto" })
    expect(Object.keys(automatic.drafters)).not.toContain(local)
    expect(withLocationEnabled(config, "Ollama", true).locations).toEqual({})
    expect(withLocationEnabled(config, "LmStudio", false).locations).toEqual({ ollama: false, lmstudio: false })
  })

  it("renames stored IDs to the ones they now name", () => {
    const config = Schema.decodeUnknownSync(CustomModelsConfigSchema)({
      hidden: ["local:old-label:9f2c1a7b3e04", "hf:owner/found/model.gguf"],
      drafters: { "local:old-label:9f2c1a7b3e04": "none" },
    })
    const renamed = withCanonicalReferences(config, (modelId) =>
      modelId.endsWith(":9f2c1a7b3e04") ? CustomFormModelIdSchema.make(local) : modelId)
    expect(renamed.hidden).toEqual([local, "hf:owner/found/model.gguf"])
    expect(Object.keys(renamed.drafters)).toEqual([local])
    expect(isHidden(renamed, CustomFormModelIdSchema.make("local:any-label:9f2c1a7b3e04"))).toBe(true)
  })

  it("sends hidden models to ICN, which lists them apart without assessing them", () => {
    expect(localModelsConfiguration(sampleConfig(), []).hidden).toEqual(["hf:owner/hidden/model.gguf"])
  })
})

describe("custom model choices", () => {
  const roots: string[] = []
  afterEach(() => { roots.splice(0).forEach((root) => rmSync(root, { recursive: true, force: true })) })

  const layer = (root: string) => {
    const globalStorage = makeGlobalStorage({ root })
    const platform = Layer.mergeAll(
      BunFileSystem.layer,
      BunPath.layer,
      Layer.succeed(GlobalStorage, GlobalStorage.of(globalStorage)),
      Layer.succeed(ProjectStorage, ProjectStorage.of(makeProjectStorage({ cwd: root }))),
      VersionLive("test"),
    )
    return CustomModelsLive.pipe(Layer.provide(Layer.merge(platform, StorageLive.pipe(Layer.provide(platform)))))
  }

  it("persists serialized writes under models and restores them after restart", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-custom-models-"))
    roots.push(root)
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const models = yield* CustomModels
      yield* Effect.all([
        models.modify((config) => Effect.succeed([undefined, { ...config, paths: [...config.paths, "/a"] }] as const)),
        models.modify((config) => Effect.succeed([undefined, { ...config, paths: [...config.paths, "/b"] }] as const)),
      ], { concurrency: "unbounded" })
    }).pipe(Effect.provide(layer(root)))))
    const restored = await Effect.runPromise(Effect.scoped(Effect.flatMap(CustomModels, (models) => models.get)
      .pipe(Effect.provide(layer(root)))))
    expect([...restored.paths].sort()).toEqual(["/a", "/b"])
    const saved = JSON.parse(await Bun.file(join(root, "config.json")).text())
    expect([...saved.models.paths].sort()).toEqual(["/a", "/b"])
  })

  it("leaves the saved choices untouched when a write's computation fails", async () => {
    const root = mkdtempSync(join(tmpdir(), "magnitude-custom-models-"))
    roots.push(root)
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const models = yield* CustomModels
      const failed = yield* models.modify(() => Effect.fail("refused" as const)).pipe(Effect.either)
      return { failed, current: yield* models.get }
    }).pipe(Effect.provide(layer(root)))))
    expect(result.failed._tag).toBe("Left")
    expect(result.current.paths).toEqual([])
  })
})
