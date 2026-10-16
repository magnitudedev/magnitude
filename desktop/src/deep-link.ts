import { Either, Option, Schema } from "effect"
import { HuggingFaceArtifactSelectorSchema, HuggingFaceRepositoryIdSchema, type HuggingFaceArtifactSelector, type HuggingFaceRepositoryId } from "@magnitudedev/sdk/desktop-host"

export const DEEP_LINK_SCHEME = "magnitude"
const MAXIMUM_LINK_LENGTH = 2048
const MAXIMUM_FILE_LENGTH = 1024
const MAXIMUM_SEGMENT_LENGTH = 255
// Hugging Face names: letters, digits, '-', '_' and '.', starting and ending with a letter or digit.
const REPOSITORY_COMPONENT = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/
const FILE_SEGMENT = /^[A-Za-z0-9._+\-()[\]@=,~]+$/

export class InvalidDeepLink extends Schema.TaggedError<InvalidDeepLink>()("InvalidDeepLink", { message: Schema.String }) {}

/** A validated `magnitude://open_from_hf?model=<owner/repo>&file=<path>` link. */
export interface HuggingFaceDeepLink {
  readonly repository: HuggingFaceRepositoryId
  readonly file: Option.Option<HuggingFaceArtifactSelector>
}

const invalid = (message: string) => Either.left(new InvalidDeepLink({ message }))

const validRepository = (value: string) => {
  const components = value.split("/")
  return components.length === 2 && components.every(component =>
    REPOSITORY_COMPONENT.test(component) && !component.includes("--") && !component.includes(".."))
    && Schema.is(HuggingFaceRepositoryIdSchema)(value)
}

const validFile = (value: string) => value.length <= MAXIMUM_FILE_LENGTH
  && value.toLowerCase().endsWith(".gguf")
  && value.split("/").every(segment => segment.length <= MAXIMUM_SEGMENT_LENGTH && segment !== "." && segment !== ".."
    && !segment.startsWith(".") && FILE_SEGMENT.test(segment))
  && Schema.is(HuggingFaceArtifactSelectorSchema)(value)

/**
 * Query values are percent-decoded without the form rule that turns '+' into a space, so a literal
 * '+' in a file name survives whether or not the sender encoded it.
 */
const parseQuery = (query: string): Either.Either<ReadonlyMap<string, string>, InvalidDeepLink> => {
  const values = new Map<string, string>()
  if (query === "") return Either.right(values)
  for (const pair of query.split("&")) {
    const separator = pair.indexOf("=")
    if (separator <= 0) return invalid("The link has a malformed parameter.")
    let key: string
    let value: string
    try {
      key = decodeURIComponent(pair.slice(0, separator))
      value = decodeURIComponent(pair.slice(separator + 1))
    } catch {
      return invalid("The link has a malformed parameter.")
    }
    if (key !== "model" && key !== "file") return invalid(`The link has an unknown parameter: ${key.slice(0, 32)}`)
    if (values.has(key)) return invalid(`The link repeats the ${key} parameter.`)
    values.set(key, value)
  }
  return Either.right(values)
}

/** Strict parsing for links that arrive from the operating system; anything unexpected is rejected. */
export const parseDeepLink = (raw: string): Either.Either<HuggingFaceDeepLink, InvalidDeepLink> => {
  if (raw.length > MAXIMUM_LINK_LENGTH) return invalid("The link is too long.")
  // Control characters and whitespace are never part of a link; URL parsing would silently strip some.
  if (/[\u0000- \u007f-\u009f]/.test(raw)) return invalid("The link contains invalid characters.")
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return invalid("The link is not a valid URL.")
  }
  if (url.protocol !== `${DEEP_LINK_SCHEME}:`) return invalid("The link is not a Magnitude link.")
  if (url.username !== "" || url.password !== "" || url.port !== "" || url.hash !== "") return invalid("The link has unexpected parts.")
  if (url.hostname !== "open_from_hf" || (url.pathname !== "" && url.pathname !== "/")) return invalid("Magnitude doesn't recognize this link.")
  const query = parseQuery(url.search.replace(/^\?/, ""))
  if (Either.isLeft(query)) return Either.left(query.left)
  const model = query.right.get("model")
  if (model === undefined) return invalid("The link doesn't name a model.")
  if (!validRepository(model)) return invalid("The link's model isn't a Hugging Face repository.")
  const file = query.right.get("file")
  if (file !== undefined && !validFile(file)) return invalid("The link's file isn't a GGUF file in the repository.")
  return Either.right({
    repository: HuggingFaceRepositoryIdSchema.make(model),
    file: file === undefined ? Option.none() : Option.some(HuggingFaceArtifactSelectorSchema.make(file)),
  })
}

export const isDeepLinkArgument = (argument: string) => argument.toLowerCase().startsWith(`${DEEP_LINK_SCHEME}:`)

/** Windows and Linux pass the link as a launch argument; macOS delivers it as an `open-url` event. */
export const deepLinkArgument = (argv: ReadonlyArray<string>): Option.Option<string> => Option.fromNullable(argv.find(isDeepLinkArgument))
