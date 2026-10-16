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
import { DrafterReferenceSchema } from "@magnitudedev/acn-protocol"
import { CustomModels, CustomModelsLive, localModelsConfiguration } from "./custom-models"

const digest = "0123456789abcdef".repeat(4)
const drafter = Schema.decodeUnknownSync(DrafterReferenceSchema)({
  _tag: "HuggingFace", repository: "z-lab/Model-DFlash", commit: "abc", path: "dflash.gguf",
})

const sampleConfig = (): CustomModelsConfig => Schema.decodeUnknownSync(CustomModelsConfigSchema)({
  declarations: {
    "hf:owner/repo-GGUF/model-Q4_K_M.gguf": {
      origin: { _tag: "Added", repository: "owner/repo-GGUF", commit: "c0ffee" },
      target: ["model-Q4_K_M.gguf"],
      projector: "mmproj.gguf",
    },
    [`local:${digest}`]: {
      origin: { _tag: "Imported", path: "/models/model.gguf", headerDigest: digest },
      target: ["/models/model.gguf"],
    },
  },
  hidden: ["hf:owner/hidden/model.gguf"],
  drafters: {
    "hf:owner/repo-GGUF/model-Q4_K_M.gguf": { _tag: "Reference", reference: drafter },
    [`local:${digest}`]: { _tag: "None" },
    "hf:owner/other/model.gguf": { _tag: "Reference", reference: { _tag: "Local", path: "/drafters/unknown.gguf" } },
  },
  folders: ["/Volumes/models"],
  disabledLocations: ["Ollama"],
})

describe("local-models configuration snapshot", () => {
  it("delivers resolved roots, declarations, and drafters with their binding method", () => {
    const configuration = localModelsConfiguration(
      sampleConfig(),
      [
        { kind: "HuggingFaceCache", path: "/hf/hub", state: "Found" },
        { kind: "Ollama", path: "/ollama", state: "NotFound" },
      ],
      (reference) => reference._tag === "HuggingFace" ? Option.some({ _tag: "DFlash" }) : Option.none(),
    )
    expect(configuration.roots).toEqual([
      { kind: "HuggingFaceCache", path: "/hf/hub", enabled: true },
      { kind: "Ollama", path: "/ollama", enabled: false },
      { kind: "Folder", path: "/Volumes/models", enabled: true },
    ])
    expect(configuration.declarations).toEqual([
      {
        modelId: "hf:owner/repo-GGUF/model-Q4_K_M.gguf",
        origin: { _tag: "Added", repository: "owner/repo-GGUF", commit: "c0ffee" },
        target: ["model-Q4_K_M.gguf"],
        projector: Option.some("mmproj.gguf"),
      },
      {
        modelId: `local:${digest}`,
        origin: { _tag: "Imported", path: "/models/model.gguf" },
        target: ["/models/model.gguf"],
        projector: Option.none(),
      },
    ])
    expect(configuration.drafters).toEqual([
      { modelId: "hf:owner/repo-GGUF/model-Q4_K_M.gguf", drafter: { _tag: "Reference", method: { _tag: "DFlash" }, reference: drafter } },
      { modelId: `local:${digest}`, drafter: { _tag: "None" } },
    ])
  })

  it("sends hidden models to ICN, which lists them apart without assessing them", () => {
    expect(localModelsConfiguration(sampleConfig(), [], () => Option.none()).hidden).toEqual(["hf:owner/hidden/model.gguf"])
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
        models.modify((config) => Effect.succeed([undefined, { ...config, folders: [...config.folders, "/a"] }] as const)),
        models.modify((config) => Effect.succeed([undefined, { ...config, folders: [...config.folders, "/b"] }] as const)),
      ], { concurrency: "unbounded" })
    }).pipe(Effect.provide(layer(root)))))
    const restored = await Effect.runPromise(Effect.scoped(Effect.flatMap(CustomModels, (models) => models.get)
      .pipe(Effect.provide(layer(root)))))
    expect([...restored.folders].sort()).toEqual(["/a", "/b"])
    const saved = JSON.parse(await Bun.file(join(root, "config.json")).text())
    expect([...saved.models.folders].sort()).toEqual(["/a", "/b"])
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
    expect(result.current.folders).toEqual([])
  })
})
