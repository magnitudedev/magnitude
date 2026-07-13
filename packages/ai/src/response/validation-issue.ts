import { ParseResult } from "effect"
import type { ValidationIssue } from "./events"

function jsonPathSegment(path: PropertyKey): string | number {
  return typeof path === "string" || typeof path === "number" ? path : String(path)
}

export function formatValidationIssue(result: ParseResult.ParseError): ValidationIssue {
  const issues = ParseResult.ArrayFormatter.formatErrorSync(result)
  if (issues.length === 0) return { path: [], message: result.message }
  return {
    path: issues[0].path.map(jsonPathSegment),
    message: issues[0].message,
  }
}
