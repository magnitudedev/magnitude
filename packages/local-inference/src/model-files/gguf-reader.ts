import { gguf } from "@huggingface/gguf"
import { Data, Effect, Schema } from "effect"
import type { SourceFileEntry } from "./types"
import { formatSchemaIssues, type SchemaIssue } from "../schema-issues"
import { GgufReaderDocument, type GgufReaderDocument as Document } from "./gguf-schema"

export class GgufForeignLibraryError extends Data.TaggedError("GgufForeignLibraryError")<{
  readonly file: SourceFileEntry["key"]
  readonly diagnostic: string
}> {}

export class GgufDocumentDecodeError extends Data.TaggedError("GgufDocumentDecodeError")<{
  readonly file: SourceFileEntry["key"]
  readonly issues: readonly SchemaIssue[]
}> {}

export interface GgufReaderApi {
  readonly read: (entry: SourceFileEntry) => Effect.Effect<Document, GgufForeignLibraryError | GgufDocumentDecodeError>
}

const boundedDiagnostic = (value: unknown): string => String(value).replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, 2048)

export const makeGgufReader = (): GgufReaderApi => ({
  read: (entry) => Effect.tryPromise({
    try: () => gguf(entry.path, { allowLocalFile: true, typedMetadata: true, computeParametersCount: true }),
    catch: (defect) => new GgufForeignLibraryError({ file: entry.key, diagnostic: boundedDiagnostic(defect) }),
  }).pipe(
    Effect.flatMap(Schema.decodeUnknown(GgufReaderDocument)),
    Effect.catchTag("ParseError", (error) => Effect.fail(
      new GgufDocumentDecodeError({ file: entry.key, issues: formatSchemaIssues(error) }),
    )),
  ),
})
