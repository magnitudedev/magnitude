import type { ParseErrorDetail, StructuralParseErrorEvent, ToolParseErrorEvent } from "../types"
import { ERROR_CATALOG, type ErrorPresentation } from "./error-catalog"
import { buildSnippet, getErrorLineFromSpan, getBlockStartLineFromSpan } from "./error-locate"

type ParseErrorEvent = StructuralParseErrorEvent | ToolParseErrorEvent

function isToolParseErrorEvent(event: ParseErrorEvent): event is ToolParseErrorEvent {
  return event._tag === "ToolParseError"
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

  const errorLine = getErrorLineFromSpan(event.error)
  const blockStartLine = getBlockStartLineFromSpan(event.error)

  const snippet = errorLine !== null
    ? buildSnippet(responseText, errorLine, strategy, blockStartLine ?? undefined)
    : ''

  const sections = [headline, ""]

  if (snippet.length > 0) {
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