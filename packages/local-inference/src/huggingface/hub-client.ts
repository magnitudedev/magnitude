import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Context, Data, Effect, Option, Redacted, Schema, pipe } from "effect"
import { Sha256Digest } from "../model-files"
import { formatSchemaIssues, type SchemaIssue } from "../schema-issues"
import {
  HuggingFaceCommitId,
  HuggingFaceObjectId,
  HuggingFaceRepositoryId,
  HuggingFaceRevision,
  type HuggingFaceCommitId as CommitId,
  type HuggingFaceObjectId as ObjectId,
  type HuggingFaceRepositoryId as RepositoryId,
  type HuggingFaceRevision as Revision,
} from "./identity"

const SearchResponse = Schema.Array(Schema.Struct({
  id: HuggingFaceRepositoryId,
  downloads: Schema.optional(Schema.NonNegativeInt),
  likes: Schema.optional(Schema.NonNegativeInt),
  lastModified: Schema.optional(Schema.String),
  tags: Schema.optional(Schema.Array(Schema.String)),
}))
const Lfs = Schema.Struct({
  oid: Schema.String.pipe(Schema.pattern(/^sha256:[a-f0-9]{64}$/)),
  size: Schema.NonNegativeInt,
})
const TreeResponse = Schema.Array(Schema.Struct({
  path: Schema.String,
  type: Schema.Literal("file", "directory"),
  size: Schema.optional(Schema.NonNegativeInt),
  oid: Schema.optional(HuggingFaceObjectId),
  lfs: Schema.optional(Lfs),
}))
type TreeResponse = Schema.Schema.Type<typeof TreeResponse>
const RevisionResponse = Schema.Struct({ sha: HuggingFaceCommitId })

export const HuggingFaceRepositoryEntryType = Schema.Literal("file", "directory")
export type HuggingFaceRepositoryEntryType = Schema.Schema.Type<typeof HuggingFaceRepositoryEntryType>
export const HuggingFaceHubOperation = Schema.Literal("search", "list-files", "resolve-revision")
export type HuggingFaceHubOperation = Schema.Schema.Type<typeof HuggingFaceHubOperation>
export const HuggingFaceHubResponseSchema = Schema.Literal("SearchResponse", "TreeResponse", "RevisionResponse")
export type HuggingFaceHubResponseSchema = Schema.Schema.Type<typeof HuggingFaceHubResponseSchema>

export interface HuggingFaceSearchResult {
  readonly repository: RepositoryId
  readonly downloads: Option.Option<number>
  readonly likes: Option.Option<number>
  readonly updatedAt: Option.Option<string>
  readonly tags: readonly string[]
}
export interface HuggingFaceRepositoryFile {
  readonly path: string
  readonly type: HuggingFaceRepositoryEntryType
  readonly sizeBytes: Option.Option<number>
  readonly oid: Option.Option<ObjectId>
  readonly lfs: Option.Option<{
    readonly sizeBytes: number
    readonly sha256: Schema.Schema.Type<typeof Sha256Digest>
  }>
}
export interface HuggingFaceRevisionResolution {
  readonly repository: RepositoryId
  readonly requested: Revision
  readonly commit: CommitId
}

interface HubErrorContext {
  readonly operation: HuggingFaceHubOperation
  readonly repository: Option.Option<RepositoryId>
}
export class HuggingFaceHubTransportError extends Data.TaggedError("HuggingFaceHubTransportError")<HubErrorContext> {}
export class HuggingFaceHubRejectedError extends Data.TaggedError("HuggingFaceHubRejectedError")<HubErrorContext & { readonly status: number }> {}
export class HuggingFaceHubInvalidResponseError extends Data.TaggedError("HuggingFaceHubInvalidResponseError")<HubErrorContext & {
  readonly schema: HuggingFaceHubResponseSchema
  readonly issues: readonly SchemaIssue[]
}> {}
export type HuggingFaceHubError = HuggingFaceHubTransportError | HuggingFaceHubRejectedError | HuggingFaceHubInvalidResponseError

export interface HuggingFaceHubClientApi {
  readonly searchModels: (query: string, limit: Option.Option<number>) => Effect.Effect<readonly HuggingFaceSearchResult[], HuggingFaceHubError>
  readonly listFiles: (repository: RepositoryId, revision: Revision) => Effect.Effect<readonly HuggingFaceRepositoryFile[], HuggingFaceHubError>
  readonly resolveRevision: (repository: RepositoryId, revision: Revision) => Effect.Effect<HuggingFaceRevisionResolution, HuggingFaceHubError>
  readonly downloadUrl: (repository: RepositoryId, commit: CommitId, path: string) => URL
}
export class HuggingFaceHubClient extends Context.Tag("@magnitudedev/local-inference/HuggingFaceHubClient")<HuggingFaceHubClient, HuggingFaceHubClientApi>() {}
export interface HuggingFaceHubClientOptions {
  readonly apiOrigin: Option.Option<URL>
  readonly token: Option.Option<Redacted.Redacted<string>>
}

interface Page<A> {
  readonly value: A
  readonly next: Option.Option<string>
}

const encodePath = (path: string): string => path.split("/").map(encodeURIComponent).join("/")
const emptyIssues: readonly SchemaIssue[] = []

const nextLink = (header: Option.Option<string>): Effect.Effect<Option.Option<string>, readonly SchemaIssue[]> => Option.match(header, {
  onNone: () => Effect.succeed(Option.none()),
  onSome: (value) => {
    const relation = value.match(/(?:^|,)\s*<([^>]+)>\s*;\s*rel="next"(?:\s*;[^,]*)?(?:,|$)/)
    const target = pipe(Option.fromNullable(relation), Option.flatMap((match) => Option.fromNullable(match[1])))
    if (Option.isSome(target)) return Effect.succeed(target)
    return value.includes("rel=\"next\"") ? Effect.fail(emptyIssues) : Effect.succeed(Option.none())
  },
})

const projectSearchResult = (item: Schema.Schema.Type<typeof SearchResponse>[number]): HuggingFaceSearchResult => ({
  repository: item.id,
  downloads: Option.fromNullable(item.downloads),
  likes: Option.fromNullable(item.likes),
  updatedAt: Option.fromNullable(item.lastModified),
  tags: Option.getOrElse(Option.fromNullable(item.tags), () => []),
})

const projectRepositoryFile = (item: TreeResponse[number]): HuggingFaceRepositoryFile => ({
  path: item.path,
  type: item.type,
  sizeBytes: Option.fromNullable(item.size),
  oid: Option.fromNullable(item.oid),
  lfs: pipe(
    Option.fromNullable(item.lfs),
    Option.map((lfs) => ({
      sizeBytes: lfs.size,
      sha256: Sha256Digest.make(lfs.oid.slice(7)),
    })),
  ),
})

export const makeHuggingFaceHubClient = (
  options: HuggingFaceHubClientOptions,
): Effect.Effect<HuggingFaceHubClientApi, never, HttpClient.HttpClient> => Effect.gen(function* () {
  const client = yield* HttpClient.HttpClient
  const origin = Option.getOrElse(options.apiOrigin, () => new URL("https://huggingface.co"))
  const token = options.token

  const requestPage = <A, I>(
    operation: HuggingFaceHubOperation,
    schemaName: HuggingFaceHubResponseSchema,
    schema: Schema.Schema<A, I>,
    route: string,
    repository: Option.Option<RepositoryId>,
  ): Effect.Effect<Page<A>, HuggingFaceHubError> => Effect.gen(function* () {
    const plainRequest = HttpClientRequest.get(new URL(route, origin).toString())
    const request = Option.match(token, {
      onNone: () => plainRequest,
      onSome: (secret) => HttpClientRequest.bearerToken(plainRequest, Redacted.value(secret)),
    })
    const response = yield* client.execute(request).pipe(
      Effect.mapError(() => new HuggingFaceHubTransportError({ operation, repository })),
    )
    if (response.status < 200 || response.status >= 300) {
      return yield* new HuggingFaceHubRejectedError({ operation, repository, status: response.status })
    }
    const body = yield* response.json.pipe(
      Effect.mapError(() => new HuggingFaceHubInvalidResponseError({ operation, repository, schema: schemaName, issues: emptyIssues })),
    )
    const value = yield* Schema.decodeUnknown(schema)(body).pipe(
      Effect.mapError((error) => new HuggingFaceHubInvalidResponseError({ operation, repository, schema: schemaName, issues: formatSchemaIssues(error) })),
    )
    const next = yield* nextLink(Option.fromNullable(response.headers["link"])).pipe(
      Effect.mapError((issues) => new HuggingFaceHubInvalidResponseError({ operation, repository, schema: schemaName, issues })),
    )
    return { value, next }
  })

  const request = <A, I>(
    operation: HuggingFaceHubOperation,
    schemaName: HuggingFaceHubResponseSchema,
    schema: Schema.Schema<A, I>,
    route: string,
    repository: Option.Option<RepositoryId>,
  ) => requestPage(operation, schemaName, schema, route, repository).pipe(
    Effect.map(({ value }) => value),
  )

  return {
    searchModels: (query, limit) => request(
      "search",
      "SearchResponse",
      SearchResponse,
      `/api/models?search=${encodeURIComponent(query)}&limit=${Option.getOrElse(limit, () => 20)}`,
      Option.none(),
    ).pipe(Effect.map((items) => items.map(projectSearchResult))),
    listFiles: (repository, revision) => Effect.gen(function* () {
      const files: HuggingFaceRepositoryFile[] = []
      const visited = new Set<string>()
      let route = Option.some(`/api/models/${repository}/tree/${encodeURIComponent(revision)}?recursive=true&expand=true`)
      while (Option.isSome(route)) {
        if (visited.has(route.value)) {
          return yield* new HuggingFaceHubInvalidResponseError({ operation: "list-files", schema: "TreeResponse", repository: Option.some(repository), issues: emptyIssues })
        }
        visited.add(route.value)
        const page: Page<TreeResponse> = yield* requestPage("list-files", "TreeResponse", TreeResponse, route.value, Option.some(repository))
        files.push(...page.value.map(projectRepositoryFile))
        route = page.next
      }
      return files
    }),
    resolveRevision: (repository, revision) => request(
      "resolve-revision",
      "RevisionResponse",
      RevisionResponse,
      `/api/models/${repository}/revision/${encodeURIComponent(revision)}`,
      Option.some(repository),
    ).pipe(Effect.map(({ sha }) => ({ repository, requested: revision, commit: sha }))),
    downloadUrl: (repository, commit, path) => new URL(`/${repository}/resolve/${commit}/${encodePath(path)}`, origin),
  }
})
