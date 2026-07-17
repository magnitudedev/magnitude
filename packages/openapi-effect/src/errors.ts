import { Data, ParseResult, Schema } from "effect";

export class OpenApiDocumentDecodeError extends Data.TaggedError(
  "OpenApiDocumentDecodeError"
)<{
  readonly cause: ParseResult.ParseError;
}> {}

export class OpenApiConfigDecodeError extends Data.TaggedError(
  "OpenApiConfigDecodeError"
)<{
  readonly cause: ParseResult.ParseError;
}> {}

export const DiagnosticRelated = Schema.Struct({
  pointer: Schema.String,
  message: Schema.String,
});
export type DiagnosticRelated = typeof DiagnosticRelated.Type;

export const Diagnostic = Schema.Struct({
  code: Schema.String,
  pointer: Schema.String,
  message: Schema.String,
  related: Schema.Array(DiagnosticRelated),
});
export type Diagnostic = typeof Diagnostic.Type;

export class OpenApiSemanticError extends Data.TaggedError(
  "OpenApiSemanticError"
)<{
  readonly diagnostics: readonly Diagnostic[];
}> {}

export class OpenApiEmitError extends Data.TaggedError("OpenApiEmitError")<{
  readonly module: string;
  readonly message: string;
}> {}

export type OpenApiEffectError =
  | OpenApiDocumentDecodeError
  | OpenApiConfigDecodeError
  | OpenApiSemanticError
  | OpenApiEmitError;
