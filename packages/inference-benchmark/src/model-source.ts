import { downloadFileToCacheDir, modelInfo } from "@huggingface/hub"
import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Option } from "effect"
import { homedir } from "node:os"
import { basename, isAbsolute, join, resolve } from "node:path"

export interface ModelProfile {
  readonly id: string
  readonly displayName: string
  readonly repository: string
  readonly revision: string
  readonly file: string
  readonly sizeBytes: number
  readonly sha256: string
  readonly contextLimit: number
}

const PROFILE_LIST: readonly ModelProfile[] = [{
  id: "qwen3.6-35b-a3b",
  displayName: "Qwen3.6 35B-A3B (UD-Q4_K_XL)",
  repository: "unsloth/Qwen3.6-35B-A3B-GGUF",
  revision: "a483e9e6cbd595906af30beda3187c2663a1118c",
  file: "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
  sizeBytes: 22_360_456_160,
  sha256: "707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450",
  contextLimit: 262_144,
}]

export const MODEL_PROFILES: ReadonlyMap<string, ModelProfile> = new Map(PROFILE_LIST.map((profile) => [profile.id, profile]))

export interface HuggingFaceModelReference {
  readonly kind: "huggingface"
  readonly id: string
  readonly repository: string
  readonly revision: string
  readonly file: string
  readonly expectedSizeBytes?: number
  readonly expectedSha256?: string
  readonly profile?: string
}

export function modelProfile(reference: string): ModelProfile | undefined {
  return MODEL_PROFILES.get(reference)
}

export interface LocalModelReference {
  readonly kind: "local"
  readonly id: string
  readonly path: string
}

export type ModelReference = HuggingFaceModelReference | LocalModelReference

export interface ResolvedModelArtifact {
  readonly id: string
  readonly path: string
  readonly source: ModelReference
  readonly cacheHit: boolean
}

export class ModelSourceError extends Data.TaggedError("ModelSourceError")<{
  readonly reference: string
  readonly operation: "resolve" | "inspect-cache" | "hub-metadata" | "download"
  readonly message: string
}> {}

function hfCacheRoot(): string {
  const configured = process.env.HF_HOME?.trim()
  return configured ? join(resolve(configured), "hub") : join(homedir(), ".cache", "huggingface", "hub")
}

function cachePointer(cacheRoot: string, source: HuggingFaceModelReference, commit: string): string {
  return join(cacheRoot, `models--${source.repository.replaceAll("/", "--")}`, "snapshots", commit, source.file)
}

function profileReference(profile: ModelProfile): HuggingFaceModelReference {
  return {
    kind: "huggingface",
    id: profile.id,
    repository: profile.repository,
    revision: profile.revision,
    file: profile.file,
    expectedSizeBytes: profile.sizeBytes,
    expectedSha256: profile.sha256,
    profile: profile.id,
  }
}

export function parseHuggingFaceUrl(reference: string): HuggingFaceModelReference | undefined {
  let url: URL
  try {
    url = new URL(reference)
  } catch {
    return undefined
  }
  if (url.protocol !== "https:" || url.hostname !== "huggingface.co") return undefined
  const segments = url.pathname.split("/").filter(Boolean).map(decodeURIComponent)
  const resolveIndex = segments.indexOf("resolve")
  if (resolveIndex !== 2 || segments.length < 5) return undefined
  const repository = `${segments[0]}/${segments[1]}`
  const revision = segments[3]!
  const file = segments.slice(4).join("/")
  return { kind: "huggingface", id: basename(file, ".gguf"), repository, revision, file }
}

export function resolveModelReference(reference: string, localPath?: string): ModelReference {
  if (localPath) return { kind: "local", id: reference, path: resolve(localPath) }
  const profile = MODEL_PROFILES.get(reference)
  if (profile) return profileReference(profile)
  const remote = parseHuggingFaceUrl(reference)
  if (remote) return remote
  if (isAbsolute(reference) || reference.startsWith("./") || reference.startsWith("../")) {
    return { kind: "local", id: basename(reference, ".gguf"), path: resolve(reference) }
  }
  throw new ModelSourceError({
    reference,
    operation: "resolve",
    message: `Unknown model '${reference}'. Use a model profile, a huggingface.co/.../resolve/... GGUF URL, or --model-path for an offline file.`,
  })
}

const fileMatches = (
  path: string,
  expectedSizeBytes?: number,
): Effect.Effect<boolean, never, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const info = yield* fs.stat(path).pipe(Effect.option)
    return Option.isSome(info)
      && info.value.type === "File"
      && (expectedSizeBytes === undefined || Number(info.value.size) === expectedSizeBytes)
  })

function accessToken(): string | undefined {
  return process.env.HF_TOKEN?.trim() || process.env.HUGGING_FACE_HUB_TOKEN?.trim() || undefined
}

export interface ResolveArtifactOptions {
  readonly reference: string
  readonly localPath?: string
  readonly cacheRoot?: string
  readonly fetch?: typeof fetch
  readonly onDownload?: (message: string) => void
}

export const resolveModelArtifact = (
  options: ResolveArtifactOptions,
): Effect.Effect<ResolvedModelArtifact, ModelSourceError, FileSystem.FileSystem> => Effect.gen(function* () {
  const source = resolveModelReference(options.reference, options.localPath)
  if (source.kind === "local") {
    if (!(yield* fileMatches(source.path))) {
      return yield* new ModelSourceError({
        reference: options.reference,
        operation: "inspect-cache",
        message: `Model file does not exist: ${source.path}`,
      })
    }
    return { id: source.id, path: source.path, source, cacheHit: true }
  }

  const token = accessToken()
  const credentials = {
    ...(token ? { accessToken: token } : {}),
    ...(options.fetch ? { fetch: options.fetch } : {}),
  }
  const commit = /^[0-9a-f]{40}$/i.test(source.revision)
    ? source.revision.toLowerCase()
    : yield* Effect.tryPromise({
        try: () => modelInfo({
          ...credentials,
          name: source.repository,
          revision: source.revision,
          additionalFields: ["sha"],
        }),
        catch: (error) => new ModelSourceError({
          reference: options.reference,
          operation: "hub-metadata",
          message: error instanceof Error ? error.message : String(error),
        }),
      }).pipe(Effect.map(({ sha }) => sha))
  const cacheRoot = options.cacheRoot ? resolve(options.cacheRoot) : hfCacheRoot()
  const cacheCandidates = options.cacheRoot
    ? [cachePointer(cacheRoot, source, commit)]
    : [
        cachePointer(join(homedir(), ".magnitude", "models"), source, commit),
        cachePointer(cacheRoot, source, commit),
      ]
  const matches = yield* Effect.forEach(cacheCandidates, (path) =>
    fileMatches(path, source.expectedSizeBytes).pipe(Effect.map((valid) => ({ path, valid }))))
  const cached = matches.find(({ valid }) => valid)
  if (cached) {
    return {
      id: source.profile ?? source.id,
      path: cached.path,
      source: { ...source, revision: commit },
      cacheHit: true,
    }
  }
  yield* Effect.sync(() => options.onDownload?.(
    `Downloading ${source.repository}/${source.file} at ${commit} to ${cacheRoot}`,
  ))
  const path = yield* Effect.tryPromise({
    try: () => downloadFileToCacheDir({
      ...credentials,
      repo: { type: "model", name: source.repository },
      revision: commit,
      path: source.file,
      cacheDir: cacheRoot,
    }),
    catch: (error) => new ModelSourceError({
      reference: options.reference,
      operation: "download",
      message: error instanceof Error ? error.message : String(error),
    }),
  })
  if (!(yield* fileMatches(path, source.expectedSizeBytes))) {
    return yield* new ModelSourceError({
      reference: options.reference,
      operation: "download",
      message: source.expectedSizeBytes === undefined
        ? "downloaded artifact is not a file"
        : `downloaded artifact size does not match ${source.expectedSizeBytes}`,
    })
  }
  return {
    id: source.profile ?? source.id,
    path,
    cacheHit: false,
    source: { ...source, revision: commit },
  }
})

export function listModelProfiles(): readonly ModelProfile[] {
  return PROFILE_LIST
}
