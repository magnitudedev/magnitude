import * as ParseResult from "effect/ParseResult"

export interface SchemaIssue {
  readonly kind: ParseResult.ArrayFormatterIssue["_tag"]
  readonly path: readonly (string | number)[]
  readonly message: string
}

export const formatSchemaIssues = (error: ParseResult.ParseError): readonly SchemaIssue[] =>
  ParseResult.ArrayFormatter.formatErrorSync(error).map((issue) => ({
    kind: issue._tag,
    path: issue.path.map((segment) => typeof segment === "number" ? segment : String(segment)),
    message: issue.message,
  }))
