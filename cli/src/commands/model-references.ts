import { homedir } from "node:os"
import { isAbsolute, join, resolve } from "node:path"
import {
  CatalogFormModelIdSchema,
  CustomFormModelIdSchema,
  HuggingFaceRepositoryIdSchema,
  ModelIdSchema,
  type CatalogFormModelId,
  type CustomFormModelId,
  type DrafterReference,
  type HuggingFaceRepositoryId,
  type ModelId,
} from "@magnitudedev/sdk"
import { Data, Effect, Option, Schema } from "effect"

export class ModelCommandError extends Data.TaggedError("ModelCommandError")<{
  readonly message: string
}> {}

const fail = (message: string) => Effect.fail(new ModelCommandError({ message }))

export const decodeCatalogId = (input: string): Effect.Effect<CatalogFormModelId, ModelCommandError> =>
  Schema.decodeUnknown(CatalogFormModelIdSchema)(input).pipe(
    Effect.mapError(() => new ModelCommandError({ message: `Invalid catalog model ID: ${input}` })),
  )

export const decodeModelId = (input: string): Effect.Effect<ModelId, ModelCommandError> =>
  Schema.decodeUnknown(ModelIdSchema)(input).pipe(
    Effect.mapError(() => new ModelCommandError({ message: `Invalid model ID: ${input}` })),
  )

const isCatalogId = Schema.is(CatalogFormModelIdSchema)
const isCustomId = Schema.is(CustomFormModelIdSchema)

/** A custom model ID; a catalog ID gets `catalogRefusal` so the person learns why instead of a syntax error. */
export const decodeCustomModelId = (
  input: string,
  catalogRefusal: (modelId: CatalogFormModelId) => string,
): Effect.Effect<CustomFormModelId, ModelCommandError> => isCatalogId(input)
  ? fail(catalogRefusal(input))
  : Schema.decodeUnknown(CustomFormModelIdSchema)(input).pipe(
      Effect.mapError(() => new ModelCommandError({ message: `Invalid model ID: ${input}` })),
    )

const HUGGING_FACE_HOSTS = new Set(["huggingface.co", "www.huggingface.co", "hf.co"])

/** A Hugging Face repository named by `owner/repo[:QUANT]` or a huggingface.co link. */
export interface HuggingFaceReference {
  readonly repository: HuggingFaceRepositoryId
  readonly revision: Option.Option<string>
  readonly version: Option.Option<string>
  readonly file: Option.Option<string>
}

const decodeRepository = (value: string): Option.Option<HuggingFaceRepositoryId> =>
  Option.getRight(Schema.decodeUnknownEither(HuggingFaceRepositoryIdSchema)(value))

const parseHuggingFaceUrl = (input: string): Option.Option<HuggingFaceReference> => {
  const url = (() => {
    try { return new URL(input) } catch { return null }
  })()
  if (url === null || (url.protocol !== "https:" && url.protocol !== "http:") || !HUGGING_FACE_HOSTS.has(url.hostname)) {
    return Option.none()
  }
  const [owner, repo, kind, revision, ...rest] = url.pathname.split("/").filter((part) => part.length > 0)
    .map(decodeURIComponent)
  if (owner === undefined || repo === undefined) return Option.none()
  const file = (kind === "blob" || kind === "resolve") && rest.length > 0 ? Option.some(rest.join("/")) : Option.none()
  return Option.map(decodeRepository(`${owner}/${repo}`), (repository) => ({
    repository,
    revision: kind !== undefined && revision !== undefined && ["tree", "blob", "resolve"].includes(kind)
      ? Option.some(revision)
      : Option.none(),
    version: Option.none(),
    file,
  }))
}

const parseRepositoryReference = (input: string): Option.Option<HuggingFaceReference> => {
  const separator = input.lastIndexOf(":")
  const repositoryPart = separator < 0 ? input : input.slice(0, separator)
  const version = separator < 0 ? Option.none<string>() : Option.some(input.slice(separator + 1))
  if (Option.exists(version, (value) => value.length === 0 || value.includes("/"))) return Option.none()
  return Option.map(decodeRepository(repositoryPart), (repository) => ({
    repository,
    revision: Option.none(),
    version,
    file: Option.none(),
  }))
}

export const parseHuggingFaceReference = (input: string): Option.Option<HuggingFaceReference> =>
  /^https?:\/\//i.test(input) ? parseHuggingFaceUrl(input) : parseRepositoryReference(input)

export type PullReference =
  | { readonly _tag: "Catalog"; readonly modelId: CatalogFormModelId }
  | { readonly _tag: "Custom"; readonly modelId: CustomFormModelId }
  | { readonly _tag: "HuggingFace"; readonly reference: HuggingFaceReference }

export const parsePullReference = (input: string): Effect.Effect<PullReference, ModelCommandError> => {
  if (isCatalogId(input)) return Effect.succeed({ _tag: "Catalog", modelId: input })
  if (isCustomId(input)) return Effect.succeed({ _tag: "Custom", modelId: input })
  return Option.match(parseHuggingFaceReference(input), {
    onNone: () => fail(`Expected a model ID, a Hugging Face link, or owner/repo[:QUANT]: ${input}`),
    onSome: (reference) => Effect.succeed({ _tag: "HuggingFace" as const, reference }),
  })
}

/** Whether a reference names a file on this computer rather than something on Hugging Face. */
export const isLocalPathReference = (input: string): boolean =>
  input.startsWith("/")
  || input.startsWith("./")
  || input.startsWith("../")
  || input.startsWith("~")
  || input.startsWith("\\")
  || /^[A-Za-z]:[\\/]/.test(input)
  || input.startsWith(".\\")
  || (input.toLowerCase().endsWith(".gguf") && input.split("/").length <= 2 && !/^https?:\/\//i.test(input))

/** Paths are sent to the service absolute, since it does not share this process's working directory. */
export const absolutePath = (input: string, cwd = process.cwd()): string =>
  input === "~" || input.startsWith("~/")
    ? join(homedir(), input.slice(1))
    : isAbsolute(input) ? input : resolve(cwd, input)

export const BUILT_IN_DRAFTER = "built-in"

export type DrafterArgument =
  | { readonly _tag: "Auto" }
  | { readonly _tag: "None" }
  | { readonly _tag: "Reference"; readonly reference: DrafterReference }
  | { readonly _tag: "Lookup"; readonly reference: string }

/** `auto`, `none`, `built-in`, a catalog model ID, or any Hugging Face or local drafter the service resolves. */
export const parseDrafterArgument = (
  input: string,
  cwd = process.cwd(),
): Effect.Effect<DrafterArgument, ModelCommandError> => {
  const value = input.trim()
  if (value.length === 0) return fail("Drafter must be auto, none, built-in, a model ID, a Hugging Face reference, or a GGUF path")
  const lower = value.toLowerCase()
  if (lower === "auto") return Effect.succeed({ _tag: "Auto" })
  if (lower === "none") return Effect.succeed({ _tag: "None" })
  if (lower === BUILT_IN_DRAFTER) return Effect.succeed({ _tag: "Reference", reference: { _tag: "Embedded" } })
  if (isCatalogId(value)) return Effect.succeed({ _tag: "Reference", reference: { _tag: "Catalog", modelId: value } })
  return Effect.succeed({
    _tag: "Lookup",
    reference: isLocalPathReference(value) ? absolutePath(value, cwd) : value,
  })
}

/** The argument that selects this drafter again with `--drafter` or `models drafter`. */
export const drafterReferenceArgument = (reference: DrafterReference): string => {
  switch (reference._tag) {
    case "Catalog": return reference.modelId
    case "Embedded": return BUILT_IN_DRAFTER
    case "HuggingFace": return `${reference.repository}/${reference.path}`
    case "Local": return reference.path
  }
}
