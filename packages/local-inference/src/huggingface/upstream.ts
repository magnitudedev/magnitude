import {
  downloadFileToCacheDir,
  listModels,
  modelInfo,
  pathsInfo,
  type ModelEntry,
  type PathInfo,
} from "@huggingface/hub"
import { Data, Effect, Option, Redacted, Stream } from "effect"
import type { HuggingFaceConnectionOptions, HuggingFaceModelSort } from "./contracts"

export interface HuggingFaceUpstreamModel extends ModelEntry {
  readonly tags: string[]
}

export type HuggingFaceUpstreamOptions = HuggingFaceConnectionOptions

export interface HuggingFaceUpstreamSearchRequest {
  readonly query: string
  readonly owner: Option.Option<string>
  readonly tags: readonly string[]
  readonly apps: readonly string[]
  readonly sort: Option.Option<HuggingFaceModelSort>
  readonly limit: Option.Option<number>
}

export interface HuggingFaceUpstreamApi {
  readonly searchModels: (request: HuggingFaceUpstreamSearchRequest) => Stream.Stream<HuggingFaceUpstreamModel, unknown>
  readonly resolveRevision: (repository: string, revision: string) => Effect.Effect<string, unknown>
  readonly pathsInfo: (repository: string, revision: string, paths: readonly string[]) => Effect.Effect<readonly PathInfo[], unknown>
  readonly downloadToCache: (request: { readonly repository: string; readonly commit: string; readonly path: string; readonly cacheDir: string }) => Effect.Effect<string, unknown>
}

export class HuggingFaceUpstreamFailure extends Data.TaggedError("HuggingFaceUpstreamFailure")<{ readonly cause: unknown }> {}

const credentials = (options: HuggingFaceUpstreamOptions) => ({
  ...(Option.isSome(options.token) ? { accessToken: Redacted.value(options.token.value) } : {}),
  ...(Option.isSome(options.hubUrl) ? { hubUrl: options.hubUrl.value.toString().replace(/\/$/, "") } : {}),
  ...(Option.isSome(options.fetch) ? { fetch: options.fetch.value } : {}),
})

const abortableFetch = (base: typeof fetch, signal: AbortSignal): typeof fetch =>
  ((input, init) => base(input, { ...init, signal: init?.signal ? AbortSignal.any([signal, init.signal]) : signal })) as typeof fetch

export const makeHuggingFaceUpstream = (options: HuggingFaceUpstreamOptions): HuggingFaceUpstreamApi => {
  const shared = credentials(options)
  return {
    searchModels: (request) => Stream.fromAsyncIterable(listModels({
      ...shared,
      search: {
        query: request.query,
        ...(Option.isSome(request.owner) ? { owner: request.owner.value } : {}),
        tags: [...request.tags],
        apps: [...request.apps],
      },
      ...(Option.isSome(request.sort) ? { sort: request.sort.value } : {}),
      ...(Option.isSome(request.limit) ? { limit: request.limit.value } : {}),
      additionalFields: ["tags"],
    }), (error) => error),
    resolveRevision: (repository, revision) => Effect.tryPromise(() => modelInfo({
      ...shared,
      name: repository,
      revision,
      additionalFields: ["sha"],
    })).pipe(Effect.map((entry) => entry.sha)),
    pathsInfo: (repository, revision, paths) => Effect.tryPromise(() => pathsInfo({
      ...shared,
      repo: { type: "model", name: repository },
      revision,
      paths: [...paths],
      expand: true,
    })),
    downloadToCache: ({ repository, commit, path, cacheDir }) => Effect.tryPromise({
      try: (signal) => downloadFileToCacheDir({
        ...shared,
        repo: { type: "model", name: repository },
        revision: commit,
        path,
        cacheDir,
        fetch: abortableFetch(Option.getOrElse(options.fetch, () => globalThis.fetch), signal),
      }),
      catch: (cause) => new HuggingFaceUpstreamFailure({ cause }),
    }),
  }
}
