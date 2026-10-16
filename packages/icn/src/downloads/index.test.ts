import { Effect, Layer, Option, Stream } from "effect"
import { describe, expect, it } from "vitest"
import type { ModelPackage, ServableModelBundle } from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"
import { IcnEvents } from "../events/index.js"
import {
  IcnDownloads,
  bundlePackages,
  makeIcnDownloads,
  packageHoldsFile,
} from "./index.js"

describe("bundle packages", () => {
  const pkg = (path: string) => ({
    id: path,
    files: [{ path }],
    source: { _tag: "HuggingFace", repository: "owner/repo", revision: "abc" },
  }) as unknown as ModelPackage

  it("lists a speculative bundle's target first, then its separate drafter", () => {
    const bundle = {
      _tag: "SpeculativeDecoding",
      target: pkg("model.gguf"),
      method: { _tag: "DFlash" },
      draftSource: { _tag: "Separate", draft: pkg("dflash.gguf") },
    } as unknown as ServableModelBundle
    expect(bundlePackages(bundle).map(({ id }) => id)).toEqual(["model.gguf", "dflash.gguf"])
    expect(packageHoldsFile(pkg("model.gguf"), "owner/repo", "abc", "model.gguf")).toBe(true)
    expect(packageHoldsFile(pkg("model.gguf"), "owner/repo", "def", "model.gguf")).toBe(false)
  })
})

describe("ICN downloads", () => {
  it("starts a bundle download for its model and rereads the download list", async () => {
    const started: unknown[] = []
    const downloads: unknown[] = []
    const client = { downloads: {
      listModelDownloads: () => Effect.sync(() => ({ downloads: [...downloads] })),
      startModelDownload: ({ payload }: { readonly payload: { readonly bundle: unknown; readonly forModel: unknown } }) => Effect.sync(() => {
        started.push(payload)
        const download = { id: "model_download_1", bundle: payload.bundle, forModel: payload.forModel, state: { _tag: "Pending", completedBytes: 0, totalBytes: 10 } }
        downloads.push(download)
        return { download: Option.some(download) }
      }),
    } } as unknown as IcnClientService
    const bundle = { _tag: "Standalone", package: { id: "p" } } as unknown as ServableModelBundle
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const service = yield* IcnDownloads
      const admitted = yield* service.start(bundle, "local:abc")
      return { admitted, listed: (yield* service.get).state.downloads.length }
    }).pipe(Effect.provide(makeIcnDownloads().pipe(Layer.provide(Layer.merge(
      Layer.succeed(IcnClient, client),
      Layer.succeed(IcnEvents, IcnEvents.of({ subscribe: Effect.succeed(Stream.never) })),
    )))))))
    expect(started).toEqual([{ bundle, forModel: Option.some("local:abc") }])
    expect(Option.isSome(result.admitted)).toBe(true)
    expect(result.listed).toBe(1)
  })
})
