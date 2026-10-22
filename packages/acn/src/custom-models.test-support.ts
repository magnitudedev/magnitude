import { Effect, Schema, SubscriptionRef } from "effect"
import {
  CustomModel,
  ModelDownload,
  type ModelDownloadEncoded,
} from "@magnitudedev/icn-protocol/schemas"
import {
  CustomFormModelIdSchema,
  DrafterReferenceSchema,
  HuggingFaceFormModelIdSchema,
  type CustomFormModelId,
  type HuggingFaceFormModelId,
} from "@magnitudedev/acn-protocol"
import { CustomModelsConfigSchema, type CustomModelsConfig } from "@magnitudedev/storage"
import type { CustomModelSource } from "./local-model-sources"
import type { CustomModelsApi } from "./custom-models"

/** A Hugging Face model ID, which is both a custom model's ID and the key config.json stores an added model under. */
const huggingFaceId = (id: string) =>
  CustomFormModelIdSchema.make(HuggingFaceFormModelIdSchema.make(id)) as CustomFormModelId & HuggingFaceFormModelId

export const addedId = huggingFaceId("hf:owner/model-GGUF/model-Q4_K_M.gguf")
export const foundId = huggingFaceId("hf:other/found-GGUF/found-Q8_0.gguf")
/** A model found in a folder, which has no Hugging Face identity. */
export const localId = CustomFormModelIdSchema.make("local:folder-model-q4_k_m:0123456789ab")

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
    // A Hugging Face cache records each file's Hub origin, so a model found there has a Hugging Face ID.
    origin: { _tag: "Found", source: { _tag: "HuggingFaceCache" } },
    files: [{ path: "/models/found-Q8_0.gguf", sizeBytes: 1_000 }],
    sizeBytes: 1_000,
    drafterPending: false,
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

/** How config.json spells `drafterReference`. */
export const drafterSetting = `hf:${drafterFiles.repository}/${drafterFiles.path}@${drafterFiles.commit}`

/** config.json's `models`, as users write it, decoded: an added model, and a drafter chosen for the found model. */
export const customModelsConfig = (overrides: Record<string, unknown> = {}): CustomModelsConfig =>
  Schema.decodeUnknownSync(CustomModelsConfigSchema)({
    added: { [addedId]: { revision: "c0ffee" } },
    drafters: { [foundId]: drafterSetting },
    ...overrides,
  })

/** A Hugging Face package of repository files at a revision, encoded; each file is weights unless it names its role. */
export const modelPackage = (
  repository: string,
  revision: string,
  files: readonly (string | { readonly path: string; readonly role: string })[],
) => {
  const described = files.map((file) => typeof file === "string" ? { path: file, role: "weights" } : file)
  return {
    id: `${repository}@${revision}:${described[0]?.path}`,
    files: described.map(({ path, role }) => ({ id: path, path, role, sha256: "a".repeat(64), sizeBytes: 10 })),
    relationships: [],
    source: { _tag: "HuggingFace", repository, revision },
    properties: { architecture: "gguf", format: "gguf", quantization: "Q4", quantizationName: "Q4" },
  }
}

const pkg = (repository: string, revision: string, path: string) => modelPackage(repository, revision, [path])

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
