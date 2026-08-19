import { Data, Effect } from "effect"
import {
  MAX_IMAGE_FILE_UPLOAD_BYTES,
  MAX_MESSAGE_UPLOAD_BYTES,
  MAX_MESSAGE_UPLOAD_COUNT,
  MAX_TEXT_FILE_UPLOAD_BYTES,
  imageMediaTypeFromFilename,
  imageMediaTypeFromMime,
  type ImageMediaType,
  type RawMessageUpload,
} from "@magnitudedev/sdk"

const TEXT_EXTENSIONS = [
  ".txt", ".md", ".mdx", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
  ".json", ".jsonc", ".yaml", ".yml", ".toml", ".xml", ".html", ".css",
  ".scss", ".less", ".py", ".rb", ".rs", ".go", ".java", ".kt", ".kts",
  ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".swift", ".sh", ".bash",
  ".zsh", ".fish", ".sql", ".graphql", ".gql", ".proto", ".env", ".ini",
  ".conf", ".config", ".properties", ".gradle", ".lua", ".php", ".r", ".dart",
  ".ex", ".exs", ".erl", ".hrl", ".fs", ".fsx", ".vue", ".svelte", ".svg",
] as const

const TEXT_MIME_TYPES = [
  "text/*",
  "application/json",
  "application/ld+json",
  "application/javascript",
  "application/xml",
  "image/svg+xml",
  "application/x-yaml",
  "application/toml",
] as const

export const MESSAGE_UPLOAD_ACCEPT = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  ...TEXT_MIME_TYPES,
  ...TEXT_EXTENSIONS,
].join(",")

export class ClientFileUploadRejected extends Data.TaggedError("ClientFileUploadRejected")<{
  readonly filename: string
  readonly reason: string
}> {}

export interface IngestedClientFile {
  readonly upload: RawMessageUpload
  readonly byteSize: number
}

export interface MessageUploadAppendResult {
  readonly uploads: RawMessageUpload[]
  readonly rejected: readonly ClientFileUploadRejected[]
}

export type ClientFileIngestResult =
  | { readonly _tag: "accepted"; readonly value: IngestedClientFile }
  | { readonly _tag: "rejected"; readonly error: ClientFileUploadRejected }

function bytesToBase64(bytes: Uint8Array): string {
  const chunkSize = 32_768
  let binary = ""
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize))
  }
  return btoa(binary)
}

function base64ByteLength(data: string): number {
  const padding = data.endsWith("==") ? 2 : data.endsWith("=") ? 1 : 0
  return Math.floor(data.length * 3 / 4) - padding
}

export function appendMessageUploads(
  existing: readonly RawMessageUpload[],
  candidates: readonly IngestedClientFile[],
): MessageUploadAppendResult {
  const uploads = [...existing]
  const rejected: ClientFileUploadRejected[] = []
  let bytes = existing.reduce((total, upload) => total + base64ByteLength(upload.data), 0)

  for (const candidate of candidates) {
    const filename = candidate.upload.type === "raw_image_clipboard"
      ? "Clipboard image"
      : candidate.upload.filename
    if (uploads.length >= MAX_MESSAGE_UPLOAD_COUNT) {
      rejected.push(new ClientFileUploadRejected({
        filename,
        reason: `A message can include at most ${MAX_MESSAGE_UPLOAD_COUNT} attachments.`,
      }))
      continue
    }
    if (bytes + candidate.byteSize > MAX_MESSAGE_UPLOAD_BYTES) {
      rejected.push(new ClientFileUploadRejected({
        filename,
        reason: "Attachments must total 25 MiB or less per message.",
      }))
      continue
    }
    uploads.push(candidate.upload)
    bytes += candidate.byteSize
  }

  return { uploads, rejected }
}

function readClientFile(file: File): Effect.Effect<Uint8Array, ClientFileUploadRejected> {
  return Effect.tryPromise({
    try: () => file.arrayBuffer().then(buffer => new Uint8Array(buffer)),
    catch: () => new ClientFileUploadRejected({
      filename: file.name,
      reason: "The file could not be read.",
    }),
  })
}

function readImageDimensions(file: File): Effect.Effect<{ width: number; height: number }, ClientFileUploadRejected> {
  return Effect.tryPromise({
    try: async () => {
      const bitmap = await createImageBitmap(file)
      try {
        return { width: bitmap.width, height: bitmap.height }
      } finally {
        bitmap.close()
      }
    },
    catch: () => new ClientFileUploadRejected({
      filename: file.name,
      reason: "The image could not be decoded.",
    }),
  })
}

function imageMediaType(file: File): ImageMediaType | null {
  return imageMediaTypeFromMime(file.type) ?? imageMediaTypeFromFilename(file.name)
}

function validateUtf8Text(filename: string, bytes: Uint8Array): Effect.Effect<void, ClientFileUploadRejected> {
  if (bytes.includes(0)) {
    return Effect.fail(new ClientFileUploadRejected({
      filename,
      reason: "Binary files are not supported.",
    }))
  }
  return Effect.try({
    try: () => {
      new TextDecoder("utf-8", { fatal: true }).decode(bytes)
    },
    catch: () => new ClientFileUploadRejected({
      filename,
      reason: "Only UTF-8 text and code files are supported.",
    }),
  })
}

export function ingestClientFile(file: File): Effect.Effect<IngestedClientFile, ClientFileUploadRejected> {
  return Effect.gen(function* () {
    const mediaType = imageMediaType(file)
    if (mediaType) {
      if (file.size > MAX_IMAGE_FILE_UPLOAD_BYTES) {
        return yield* new ClientFileUploadRejected({
          filename: file.name,
          reason: "Images must be 10 MiB or smaller.",
        })
      }
      const [bytes, dimensions] = yield* Effect.all([
        readClientFile(file),
        readImageDimensions(file),
      ], { concurrency: "unbounded" })
      return {
        byteSize: bytes.byteLength,
        upload: {
          type: "raw_image_file",
          data: bytesToBase64(bytes),
          filename: file.name,
          mediaType,
          width: dimensions.width,
          height: dimensions.height,
        },
      }
    }

    if (file.size > MAX_TEXT_FILE_UPLOAD_BYTES) {
      return yield* new ClientFileUploadRejected({
        filename: file.name,
        reason: "Text and code files must be 500 KiB or smaller.",
      })
    }

    const bytes = yield* readClientFile(file)
    if (bytes.byteLength === 0) {
      return yield* new ClientFileUploadRejected({
        filename: file.name,
        reason: "Empty files cannot be attached.",
      })
    }
    yield* validateUtf8Text(file.name, bytes)

    return {
      byteSize: bytes.byteLength,
      upload: {
        type: "raw_text_file",
        data: bytesToBase64(bytes),
        filename: file.name,
      },
    }
  })
}

export function ingestClientFiles(files: readonly File[]): Effect.Effect<readonly ClientFileIngestResult[]> {
  return Effect.forEach(
    files,
    file => ingestClientFile(file).pipe(
      Effect.match({
        onFailure: error => ({ _tag: "rejected" as const, error }),
        onSuccess: value => ({ _tag: "accepted" as const, value }),
      }),
    ),
    { concurrency: 4 },
  )
}
