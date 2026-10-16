import { Either, Option, Schema } from "effect"
import { posix, win32 } from "node:path"
import { HuggingFaceRepositoryIdSchema, type HuggingFaceRepositoryId } from "@magnitudedev/acn-protocol"
import { expandPath, type DiscoveryEnvironment } from "./icn/discovery-roots"

/** What a pasted link, `owner/repo[:QUANT]`, `owner/repo/file.gguf`, or local path names. */
export type ModelReference =
  | {
      readonly _tag: "HuggingFace"
      readonly repository: HuggingFaceRepositoryId
      readonly revision: string
      readonly file: Option.Option<string>
      readonly quant: Option.Option<string>
    }
  | { readonly _tag: "Local"; readonly path: string }

const HUGGING_FACE_HOSTS = new Set(["huggingface.co", "www.huggingface.co", "hf.co"])

const repository = (owner: string | undefined, name: string | undefined): Either.Either<HuggingFaceRepositoryId, string> =>
  Schema.decodeUnknownEither(HuggingFaceRepositoryIdSchema)(`${owner ?? ""}/${name ?? ""}`).pipe(
    Either.mapLeft(() => "A Hugging Face repository is named owner/repository"),
  )

const isLocalPath = (value: string) => value.startsWith("/") || value.startsWith("~")
  || value.startsWith("./") || value.startsWith("../") || value.startsWith(".\\") || value.startsWith("..\\")
  || win32.isAbsolute(value) && /^([a-zA-Z]:[\\/]|\\\\)/.test(value)

const fromUrl = (input: string): Either.Either<ModelReference, string> => {
  let url: URL
  try {
    url = new URL(input)
  } catch {
    return Either.left("This link isn't a valid URL")
  }
  if (!HUGGING_FACE_HOSTS.has(url.hostname)) return Either.left("Only huggingface.co links can be added")
  const [owner, name, kind, revision, ...rest] = url.pathname.split("/").filter((part) => part.length > 0)
    .map((part) => decodeURIComponent(part))
  return Either.map(repository(owner, name), (id): ModelReference => {
    const pinned = (kind === "tree" || kind === "blob" || kind === "resolve") && revision !== undefined
    const file = (kind === "blob" || kind === "resolve") && rest.length > 0 ? Option.some(rest.join("/")) : Option.none()
    return { _tag: "HuggingFace", repository: id, revision: pinned ? revision : "main", file, quant: Option.none() }
  })
}

/**
 * Parses a model or drafter reference. A path is local only when it is absolute or starts with
 * `~`, `./`, or `../`; anything else is Hugging Face syntax, so `owner/repo/file.gguf` names a file
 * in a repository.
 */
export const parseModelReference = (
  input: string,
  environment: DiscoveryEnvironment,
): Either.Either<ModelReference, string> => {
  const value = input.trim()
  if (value.length === 0) return Either.left("Enter a Hugging Face link, owner/repository, or a GGUF file path")
  if (/^https?:\/\//i.test(value)) return fromUrl(value)
  if (/^(www\.)?(huggingface\.co|hf\.co)\//i.test(value)) return fromUrl(`https://${value}`)
  if (isLocalPath(value)) {
    const path = expandPath(environment, value)
    return Either.right({ _tag: "Local", path: (environment.platform === "win32" ? win32 : posix).normalize(path) })
  }
  const [head, quant] = value.split(":") as [string, string | undefined]
  if (quant !== undefined && (quant.length === 0 || quant.includes("/"))) {
    return Either.left("A version is named owner/repository:QUANT")
  }
  const [owner, name, ...file] = head.split("/")
  return Either.map(repository(owner, name), (id): ModelReference => ({
    _tag: "HuggingFace",
    repository: id,
    revision: "main",
    file: file.length > 0 ? Option.some(file.join("/")) : Option.none(),
    quant: Option.fromNullable(quant),
  }))
}

/**
 * A drafter the user entered, in the form the inference service reads: `owner/repo[@revision]:path`
 * for a repository file (from a link, `owner/repo/path.gguf`, or `hf:owner/repo/path.gguf`), an
 * absolute path for a local file, and anything else as typed for the service to judge.
 */
export const drafterInput = (input: string, environment: DiscoveryEnvironment): string => {
  const value = input.trim().replace(/^hf:/i, "")
  return Either.match(parseModelReference(value, environment), {
    onLeft: () => value,
    onRight: (reference) => {
      if (reference._tag === "Local") return reference.path
      if (Option.isSome(reference.quant) || Option.isNone(reference.file)) return value
      const revision = reference.revision === "main" ? "" : `@${reference.revision}`
      return `${reference.repository}${revision}:${reference.file.value}`
    },
  })
}

/** A repository's display name: its name without a trailing GGUF marker, words split at separators. */
export const repositoryDisplayName = (id: string): string => {
  const name = id.split("/").at(-1) ?? id
  const trimmed = name.replace(/[-_.]?gguf$/i, "")
  return (trimmed.length > 0 ? trimmed : name).replace(/[-_]+/g, " ").trim()
}
