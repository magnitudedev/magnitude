import { HubApiError, InvalidApiResponseFormatError } from "@huggingface/hub"
import { Effect, Layer, Option, Schema, Stream } from "effect"
import { Sha256Digest } from "../model-files"
import {
  HuggingFaceArtifact,
  HuggingFaceArtifactRequest,
  type HuggingFaceConnectionOptions,
  HuggingFaceGitContent,
  HuggingFaceHub,
  type HuggingFaceHubApi,
  HuggingFaceLfsContent,
  HuggingFaceModelSummary,
  HuggingFaceSearchRequest,
  HuggingFaceXetContent,
} from "./contracts"
import { makeHuggingFaceArtifactId } from "./artifact-identity"
import {
  HuggingFaceAccessDeniedError,
  HuggingFaceArtifactInvalidError,
  HuggingFaceAuthenticationError,
  type HuggingFaceHubError,
  type HuggingFaceHubOperation,
  HuggingFaceInvalidResponseError,
  HuggingFaceInvalidRequestError,
  HuggingFaceNotFoundError,
  HuggingFaceRateLimitedError,
  HuggingFaceUnavailableError,
} from "./errors"
import {
  HuggingFaceCommitId,
  HuggingFaceFilePath,
  HuggingFaceObjectId,
  HuggingFaceRepositoryId,
  type HuggingFaceRevision,
  HuggingFaceXetHash,
} from "./identity"
import { HuggingFaceUpstreamFailure, makeHuggingFaceUpstream, type HuggingFaceUpstreamApi } from "./upstream"

interface ErrorContext {
  readonly operation: HuggingFaceHubOperation
  readonly repository: Option.Option<HuggingFaceRepositoryId>
  readonly revision?: Option.Option<HuggingFaceRevision>
  readonly path?: Option.Option<Schema.Schema.Type<typeof HuggingFaceFilePath>>
}

const diagnostic = (error: unknown): string => error instanceof Error ? error.name.slice(0, 128) : "UnknownError"

export const mapHuggingFaceHubError = (input: unknown, context: ErrorContext): HuggingFaceHubError => {
  const error = input instanceof HuggingFaceUpstreamFailure ? input.cause : input
  if (error instanceof HubApiError) {
    if (error.statusCode === 401) return new HuggingFaceAuthenticationError(context)
    if (error.statusCode === 403) return new HuggingFaceAccessDeniedError(context)
    if (error.statusCode === 404) return new HuggingFaceNotFoundError({ ...context, revision: context.revision ?? Option.none(), path: context.path ?? Option.none() })
    if (error.statusCode === 429) return new HuggingFaceRateLimitedError(context)
    return new HuggingFaceUnavailableError({ ...context, status: Option.some(error.statusCode), diagnostic: `HubApiError:${error.statusCode}` })
  }
  if (error instanceof InvalidApiResponseFormatError) return new HuggingFaceInvalidResponseError({ ...context, diagnostic: diagnostic(error) })
  return new HuggingFaceUnavailableError({ ...context, status: Option.none(), diagnostic: diagnostic(error) })
}

const invalid = (request: HuggingFaceArtifactRequest, reason: HuggingFaceArtifactInvalidError["reason"], path = Option.none<string>()) =>
  new HuggingFaceArtifactInvalidError({ repository: request.repository, reason, path })

export const makeHuggingFaceHubFromUpstream = (upstream: HuggingFaceUpstreamApi): HuggingFaceHubApi => {
  const searchModels = (request: Schema.Schema.Type<typeof HuggingFaceSearchRequest>) => upstream.searchModels(request).pipe(
    Stream.mapError((error) => mapHuggingFaceHubError(error, { operation: "search", repository: Option.none() })),
    Stream.mapEffect((entry) => Schema.decodeUnknown(HuggingFaceRepositoryId)(entry.name).pipe(
      Effect.flatMap((repository) => Schema.validate(HuggingFaceModelSummary)({ repository, private: entry.private, gated: entry.gated, downloads: entry.downloads, likes: entry.likes, updatedAt: entry.updatedAt, tags: entry.tags ?? [] })),
      Effect.mapError(() => new HuggingFaceInvalidResponseError({ operation: "search", repository: Option.none(), diagnostic: "InvalidModelSummary" })),
    )),
  )
  return {
    searchModels: (input) => Stream.unwrap(Schema.validate(HuggingFaceSearchRequest)(input).pipe(
      Effect.map(searchModels),
      Effect.mapError(() => new HuggingFaceInvalidRequestError({ operation: "search", diagnostic: "InvalidSearchRequest" })),
    )),
    resolveArtifact: (input) => Effect.gen(function* () {
      const request = yield* Schema.validate(HuggingFaceArtifactRequest)(input).pipe(
        Effect.mapError(() => new HuggingFaceInvalidRequestError({ operation: "resolve-paths", diagnostic: "InvalidArtifactRequest" })),
      )
      if (request.files.length === 0) return yield* invalid(request, "empty-files")
      const decodedFiles: Array<{ readonly path: Schema.Schema.Type<typeof HuggingFaceFilePath>; readonly role: typeof request.files[number]["role"]; readonly shardIndex: typeof request.files[number]["shardIndex"] }> = []
      for (const file of request.files) {
        const decoded = yield* Schema.decodeUnknown(HuggingFaceFilePath)(file.path).pipe(Effect.mapError(() => invalid(request, "unsafe-path", Option.some(file.path))))
        decodedFiles.push({ ...file, path: decoded })
      }
      if (new Set(decodedFiles.map(({ path }) => path)).size !== decodedFiles.length) return yield* invalid(request, "duplicate-file")
      const decodedPaths = new Map<string, Schema.Schema.Type<typeof HuggingFaceFilePath>>(decodedFiles.map(({ path }) => [path, path]))
      if (request.relationships.some(({ fromPath, toPath }) => !decodedPaths.has(fromPath) || !decodedPaths.has(toPath))) return yield* invalid(request, "invalid-relationship")

      const commitText = yield* upstream.resolveRevision(request.repository, request.revision).pipe(
        Effect.mapError((error) => mapHuggingFaceHubError(error, { operation: "resolve-revision", repository: Option.some(request.repository), revision: Option.some(request.revision) })),
      )
      const commit = yield* Schema.decodeUnknown(HuggingFaceCommitId)(commitText).pipe(
        Effect.mapError(() => new HuggingFaceInvalidResponseError({ operation: "resolve-revision", repository: Option.some(request.repository), diagnostic: "InvalidCommitId" })),
      )
      const remoteFiles = yield* upstream.pathsInfo(request.repository, commit, decodedFiles.map(({ path }) => path)).pipe(
        Effect.mapError((error) => mapHuggingFaceHubError(error, { operation: "resolve-paths", repository: Option.some(request.repository), revision: Option.some(request.revision) })),
      )
      const byPath = new Map(remoteFiles.map((file) => [file.path, file]))
      const files: HuggingFaceArtifact["files"][number][] = []
      for (const requested of decodedFiles) {
        const remote = byPath.get(requested.path)
        if (!remote || remote.type !== "file") return yield* invalid(request, "missing-file", Option.some(requested.path))
        const identity = yield* Effect.gen(function* () {
          if (remote.lfs?.oid) {
            const value = remote.lfs.oid.replace(/^sha256:/, "").toLowerCase()
            return new HuggingFaceLfsContent({ sha256: yield* Schema.decodeUnknown(Sha256Digest)(value) })
          }
          if (remote.xetHash) return new HuggingFaceXetContent({ hash: yield* Schema.decodeUnknown(HuggingFaceXetHash)(remote.xetHash.toLowerCase()) })
          if (remote.oid) return new HuggingFaceGitContent({ oid: yield* Schema.decodeUnknown(HuggingFaceObjectId)(remote.oid.toLowerCase()) })
          return yield* invalid(request, "invalid-content-identity", Option.some(requested.path))
        }).pipe(Effect.mapError(() => invalid(request, "invalid-content-identity", Option.some(requested.path))))
        files.push({ ...requested, sizeBytes: remote.lfs?.size ?? remote.size, content: identity })
      }
      const relationships = request.relationships.map((relationship) => ({
        kind: relationship.kind,
        fromPath: decodedPaths.get(relationship.fromPath)!,
        toPath: decodedPaths.get(relationship.toPath)!,
      }))
      const identityInput = { repository: request.repository, commit, files, relationships }
      const candidate = { id: makeHuggingFaceArtifactId(identityInput), requestedRevision: request.revision, ...identityInput, totalBytes: files.reduce((sum, file) => sum + file.sizeBytes, 0) }
      return yield* Schema.validate(HuggingFaceArtifact)(candidate).pipe(
        Effect.mapError(() => invalid(request, "invalid-artifact")),
      )
    }),
  }
}

export const makeHuggingFaceHub = (options: HuggingFaceConnectionOptions): HuggingFaceHubApi =>
  makeHuggingFaceHubFromUpstream(makeHuggingFaceUpstream(options))

export const HuggingFaceHubLive = (options: HuggingFaceConnectionOptions): Layer.Layer<HuggingFaceHub> =>
  Layer.succeed(HuggingFaceHub, makeHuggingFaceHub(options))
