import { Effect, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelDownloadIdSchema,
  ModelPackageIdSchema,
  type ModelBundleDownload,
  type ModelPackagesState,
  type ServableModelBundle,
} from "@magnitudedev/acn-protocol"
import { ProviderModelIdSchema } from "@magnitudedev/sdk"
import { LocalModelPackages } from "./local-model-packages"
import { LocalModelSyncs, LocalModelSyncsLive } from "./local-model-syncs"

const bundle: ServableModelBundle = {
  _tag: "Standalone",
  package: {
    id: ModelPackageIdSchema.make("shared-package"),
    source: { _tag: "Local", path: "/models/shared.gguf" },
    files: [],
    relationships: [],
    properties: {
      format: "gguf",
      quantization: "Q4_K_M",
      quantizationName: "4-bit",
      architecture: "test",
      maximumContextLength: Option.none(),
      intrinsicModelId: Option.none(),
      intrinsicQualityId: Option.none(),
    },
  },
}

const download = (id: string): ModelBundleDownload => ({
  id: ModelDownloadIdSchema.make(id),
  bundle,
  state: {
    _tag: "Downloading",
    stage: "downloading",
    completedBytes: 1,
    totalBytes: 2,
    bytesPerSecond: Option.none(),
  },
})

describe("local model sync correlation", () => {
  it("keeps model operations distinct when their ICN occurrences have the same bundle", async () => {
    const first = download("download-first")
    const second = download("download-second")
    const packageState: ModelPackagesState = {
      inventory: { _tag: "Ready" },
      entries: [],
      downloads: [first, second],
    }
    const packages = LocalModelPackages.of({
      initialized: Effect.succeed(true),
      state: Effect.succeed(packageState),
      changes: Stream.empty,
      installedPackageIds: Effect.succeed(new Set()),
      refresh: Effect.void,
    })

    await Effect.runPromise(Effect.gen(function* () {
      const syncs = yield* LocalModelSyncs
      const firstModel = ProviderModelIdSchema.make("catalog:first")
      const secondModel = ProviderModelIdSchema.make("catalog:second")
      yield* syncs.admitted(firstModel, first.id)
      yield* syncs.admitted(secondModel, second.id)

      expect(yield* syncs.download(firstModel)).toEqual(Option.some(first))
      expect(yield* syncs.download(secondModel)).toEqual(Option.some(second))

      yield* syncs.current(firstModel)
      expect(yield* syncs.download(firstModel)).toEqual(Option.none())
      expect(yield* syncs.download(secondModel)).toEqual(Option.some(second))
    }).pipe(
      Effect.provide(LocalModelSyncsLive),
      Effect.provideService(LocalModelPackages, packages),
    ))
  })
})
