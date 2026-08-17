import { gguf } from "@huggingface/gguf"
import * as FileSystem from "@effect/platform/FileSystem"
import { Data, Effect, Stream } from "effect"
import type { ModelIdentity } from "./domain"
import { sha256 } from "./hash"

export class ModelIdentityError extends Data.TaggedError("ModelIdentityError")<{
  readonly path: string
  readonly operation: "inspect" | "hash" | "validate"
  readonly message: string
}> {}

export interface ResolveModelOptions {
  readonly id: string
  readonly artifactPath: string
  readonly verifiedArtifactSha256?: string
  readonly maxContextTokens?: number
}

function metadataValue(
  metadata: Record<string, { readonly value?: unknown }>,
  key: string,
): unknown {
  return metadata[key]?.value
}

export const hashFileSha256 = (
  path: string,
): Effect.Effect<string, ModelIdentityError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const hasher = new Bun.CryptoHasher("sha256")
    yield* fs.stream(path).pipe(
      Stream.runForEach((chunk) => Effect.sync(() => { hasher.update(chunk) })),
      Effect.mapError((error) => new ModelIdentityError({
        path,
        operation: "hash",
        message: error instanceof Error ? error.message : String(error),
      })),
    )
    return hasher.digest("hex")
  })

export const resolveModelIdentity = (
  options: ResolveModelOptions,
): Effect.Effect<ModelIdentity, ModelIdentityError, FileSystem.FileSystem> =>
  Effect.gen(function* () {
    const document = yield* Effect.tryPromise({
      try: () => gguf(options.artifactPath, { allowLocalFile: true, typedMetadata: true }),
      catch: (error) => new ModelIdentityError({
        path: options.artifactPath,
        operation: "inspect",
        message: error instanceof Error ? error.message : String(error),
      }),
    })
    const metadata = document.typedMetadata as Record<string, { readonly value?: unknown }>
    const architecture = metadataValue(metadata, "general.architecture")
    const trained = typeof architecture === "string"
      ? metadataValue(metadata, `${architecture}.context_length`)
      : undefined
    const trainedContext = typeof trained === "bigint"
      ? Number(trained)
      : typeof trained === "number" ? trained : undefined
    if (!trainedContext || !Number.isSafeInteger(trainedContext) || trainedContext <= 0) {
      return yield* new ModelIdentityError({
        path: options.artifactPath,
        operation: "validate",
        message: "GGUF does not declare a valid architecture context length",
      })
    }
    const template = metadataValue(metadata, "tokenizer.chat_template")
    if (typeof template !== "string" || template.length === 0) {
      return yield* new ModelIdentityError({
        path: options.artifactPath,
        operation: "validate",
        message: "GGUF does not declare a chat template",
      })
    }
    return {
      id: options.id,
      artifactPath: options.artifactPath,
      artifactSha256: options.verifiedArtifactSha256 ?? (yield* hashFileSha256(options.artifactPath)),
      contextLimit: options.maxContextTokens && options.maxContextTokens > 0
        ? Math.min(trainedContext, options.maxContextTokens)
        : trainedContext,
      chatTemplateDigest: sha256(template),
    }
  })
