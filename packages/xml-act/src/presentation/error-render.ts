import type { ParseErrorDetail, StructuralParseErrorEvent, ToolParseErrorEvent } from "../types"
import { ERROR_CATALOG, type ErrorPresentation } from "./error-catalog"
import { buildSnippet, findErrorLine } from "./error-locate"

type ParseErrorEvent = StructuralParseErrorEvent | ToolParseErrorEvent

function isToolParseErrorEvent(event: ParseErrorEvent): event is ToolParseErrorEvent {
  return event._tag === "ToolParseError"
}

function getAnchor(event: ParseErrorEvent): string | null {
  const { error } = event

  if ("raw" in error && typeof error.raw === "string" && error.raw.length > 0) {
    return error.raw
  }

  switch (error._tag) {
    case "StrayCloseTag":
      return `</${error.tagName}>`
    case "UnknownTool":
      return `<magnitude:invoke tool="${error.tagName}">`
    case "MalformedTag":
      return error.tagName ? `<${error.tagName}` : "<"
    case "MissingToolName":
      return "<magnitude:invoke"
    case "UnexpectedContent":
      return error.context
    case "UnclosedThink":
      return "<magnitude:reason"
    case "UnknownParameter":
    case "DuplicateParameter":
    case "MissingRequiredField":
    case "SchemaCoercionError":
    case "JsonStructuralError":
    case "IncompleteTool":
      return `<magnitude:invoke tool="${error.tagName}">`
    default:
      return null
  }
}

function getBlockStartLine(event: ParseErrorEvent, responseText: string, errorLine: number | null): number | undefined {
  const { error } = event

  if (error._tag === "UnclosedThink") {
    return findErrorLine(responseText, "<magnitude:reason") ?? undefined
  }

  if (
    error._tag === "UnknownParameter" ||
    error._tag === "DuplicateParameter" ||
    error._tag === "MissingRequiredField" ||
    error._tag === "SchemaCoercionError" ||
    error._tag === "JsonStructuralError" ||
    error._tag === "IncompleteTool"
  ) {
    return findErrorLine(responseText, `<magnitude:invoke tool="${error.tagName}">`) ?? errorLine ?? undefined
  }

  return undefined
}

function buildFallbackSnippet(responseText: string): string {
  const lines = responseText.split("\n").slice(0, 5)
  return lines.map((line, index) => `${index + 1}|${line}`).join("\n")
}

function renderToolSection(event: ToolParseErrorEvent): string {
  const lines = [`Tool: ${event.tagName}`]

  if (event.correctToolShape) {
    lines.push("Expected:")
    lines.push(event.correctToolShape)
  }

  return lines.join("\n")
}

function renderHints(hints: readonly string[]): string {
  return ["Hints:", ...hints.map(hint => `- ${hint}`)].join("\n")
}

export function renderParseError(
  event: StructuralParseErrorEvent | ToolParseErrorEvent,
  responseText: string,
): string {
  const tag = event.error._tag as ParseErrorDetail['_tag']
  const catalog = ERROR_CATALOG[tag] as ErrorPresentation<ParseErrorDetail>
  const headline = catalog.headline(event.error)
  const hints = catalog.hints(event.error)
  const strategy = catalog.snippetStrategy

  const anchor = getAnchor(event)
  const errorLine = anchor ? findErrorLine(responseText, anchor) : null

  const snippet =
    errorLine === null
      ? buildFallbackSnippet(responseText)
      : buildSnippet(responseText, errorLine, strategy, getBlockStartLine(event, responseText, errorLine))

  const sections = [headline, ""]

  if (errorLine === null) {
    sections.push("Location: unable to isolate exact line.")
    if (snippet.length > 0) {
      sections.push("")
      sections.push(snippet)
    }
  } else if (snippet.length > 0) {
    sections.push(snippet)
  }

  if (isToolParseErrorEvent(event)) {
    sections.push("")
    sections.push(renderToolSection(event))
  }

  if (hints.length > 0) {
    sections.push("")
    sections.push(renderHints(hints))
  }

  return `<parse_error>\n${sections.join("\n")}\n</parse_error>`
}
