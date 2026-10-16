import { Effect, Schema, SubscriptionRef } from "effect"
import {
  CustomModel,
  ModelDownload,
  type ModelDownloadEncoded,
} from "@magnitudedev/icn-protocol/schemas"
import { CustomFormModelIdSchema, DrafterReferenceSchema } from "@magnitudedev/acn-protocol"
import { CustomModelsConfigSchema, type CustomModelsConfig } from "@magnitudedev/storage"
import type { CustomModelSource } from "./local-model-sources"
import type { CustomModelsApi } from "./custom-models"

export const headerDigest = "0123456789abcdef".repeat(4)
export const addedId = CustomFormModelIdSchema.make("hf:owner/model-GGUF/model-Q4_K_M.gguf")
export const foundId = CustomFormModelIdSchema.make("hf:other/found-GGUF/found-Q8_0.gguf")
export const importedId = CustomFormModelIdSchema.make(`local:${headerDigest}`)

export const readyModel = {
  metadata: {
    format: "gguf",
    architecture: "qwen35moe",
    quantization: "q4_k_m",
    quantizationName: "Q4_K_M",
    storageBytes: 1_000,
  },
  profile: { contextLength: 32_768 },
}

export const nativeCustomModel = (overrides: Readonly<Record<string, unknown>> = {}): CustomModel =>
  Schema.decodeUnknownSync(CustomModel)({
    modelId: foundId,
    displayName: "Found model",
    origin: { _tag: "Found", source: { _tag: "LmStudio" } },
    files: [{ path: "/models/found-Q8_0.gguf", sizeBytes: 1_000 }],
    sizeBytes: 1_000,
    profile: { contextLength: 32_768 },
    state: {
      _tag: "Ready",
      catalogAttribution: { _tag: "NotInCatalog" },
      installation: { _tag: "Resolved", installedBytes: 1_000, ownership: "External", primaryPath: "/models/found-Q8_0.gguf" },
      model: readyModel,
    },
    ...overrides,
  })

export const customSource = (model: CustomModel): CustomModelSource => {
  const { modelId, ...source } = model
  return { id: Schema.decodeUnknownSync(CustomFormModelIdSchema)(modelId), source }
}

const drafterFiles = { repository: "magnitudedev/model-DFlash-GGUF", commit: "d1", path: "dflash.gguf" } as const

export const drafterReference = Schema.decodeUnknownSync(DrafterReferenceSchema)({ _tag: "HuggingFace", ...drafterFiles })

export const customModelsConfig = (overrides: Record<string, unknown> = {}): CustomModelsConfig =>
  Schema.decodeUnknownSync(CustomModelsConfigSchema)({
    declarations: {
      [addedId]: {
        origin: { _tag: "Added", repository: "owner/model-GGUF", commit: "c0ffee" },
        target: ["model-Q4_K_M.gguf"],
      },
      [importedId]: {
        origin: { _tag: "Imported", path: "/models/imported.gguf", headerDigest },
        target: ["/models/imported.gguf"],
      },
    },
    drafters: { [foundId]: { _tag: "Reference", reference: drafterReference } },
    ...overrides,
  })

const pkg = (repository: string, revision: string, path: string) => ({
  id: `${repository}@${revision}:${path}`,
  files: [{ id: path, path, role: "weights", sha256: "a".repeat(64), sizeBytes: 10 }],
  relationships: [],
  source: { _tag: "HuggingFace", repository, revision },
  properties: { architecture: "gguf", format: "gguf", quantization: "Q4", quantizationName: "Q4" },
})

export const nativeDownload = (
  id: string,
  forModel: string,
  files: { readonly repository: string; readonly revision: string; readonly path: string },
  state: ModelDownloadEncoded["state"],
): ModelDownload => Schema.decodeUnknownSync(ModelDownload)({
  id,
  bundle: { _tag: "Standalone", package: pkg(files.repository, files.revision, files.path) },
  forModel,
  state,
})

export const targetDownload = (id: string, state: ModelDownloadEncoded["state"]) =>
  nativeDownload(id, addedId, { repository: "owner/model-GGUF", revision: "c0ffee", path: "model-Q4_K_M.gguf" }, state)

/** A drafter fetched alone for the found model; its package may come from any repository, the catalog's included. */
export const drafterDownload = (id: string, state: ModelDownloadEncoded["state"], forModel: string = foundId) =>
  nativeDownload(id, forModel, { repository: drafterFiles.repository, revision: drafterFiles.commit, path: drafterFiles.path }, state)

/** An in-memory CustomModels with the same serialized read-modify-write contract. */
export const makeTestCustomModels = (initial: CustomModelsConfig): Effect.Effect<CustomModelsApi> => Effect.gen(function* () {
  const state = yield* SubscriptionRef.make(initial)
  const lock = yield* Effect.makeSemaphore(1)
  return {
    get: SubscriptionRef.get(state),
    changes: state.changes,
    modify: (f) => lock.withPermits(1)(Effect.gen(function* () {
      const [result, next] = yield* f(yield* SubscriptionRef.get(state))
      yield* SubscriptionRef.set(state, next)
      return result
    })),
  }
})
