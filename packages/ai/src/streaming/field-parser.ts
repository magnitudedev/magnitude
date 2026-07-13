/**
 * StreamingFieldParser<A> — the public API for streaming JSON field parsing.
 *
 * Combines:
 * - Incremental JSON parsing (json-parser.ts)
 * - Snapshot diffing → FieldEvent production (from codec walkAndDiff)
 * - Schema validation (progressive + final)
 * - Typed partial access via StreamingPartial<A>
 *
 * Never-switching generic: erased when no schema, concrete when schema provided.
 */

import { type ParseResult, Schema } from "effect"
import type { ValidationIssue } from "../response/events"
import { formatValidationIssue } from "../response/validation-issue"
import { createIncrementalJsonParser } from "./parser"
import type { FieldEvent, ParsedValue, StreamingPartial } from "./types"
import { parsedValueToJson, parsedValueToStreamingPartial } from "./values"
import { deriveStreamingSchema } from "./streaming-schema"
import type { StreamingSchemaResult } from "./streaming-schema"

// ---------------------------------------------------------------------------
// Public interfaces
// ---------------------------------------------------------------------------

export interface StreamingFieldParserErased {
  push(chunk: string): readonly FieldEvent[]
  end(): readonly FieldEvent[]
  readonly partial: StreamingPartial<Record<string, unknown>> | undefined
  readonly decoded: Record<string, unknown> | null
  readonly valid: boolean
  readonly validationIssue: ValidationIssue | null
}

export interface StreamingFieldParserConcrete<A> {
  push(chunk: string): readonly FieldEvent[]
  end(): readonly FieldEvent[]
  readonly partial: StreamingPartial<A>
  readonly decoded: A | null
  readonly valid: boolean
  readonly validationIssue: ValidationIssue | null
}

export type StreamingFieldParser<A = never> = [A] extends [never]
  ? StreamingFieldParserErased
  : StreamingFieldParserConcrete<A>

// ---------------------------------------------------------------------------
// Snapshot diffing → FieldEvent production
// ---------------------------------------------------------------------------

interface FieldState {
  seenText: string
  complete: boolean
}

type Decoded<A> =
  | { readonly _tag: "Undecoded" }
  | { readonly _tag: "Decoded"; readonly value: A }

type ParserValidation<A> =
  | { readonly _tag: "Valid"; readonly decoded: Decoded<A> }
  | { readonly _tag: "Invalid"; readonly issue: ValidationIssue }

const UNDECODED: Decoded<never> = { _tag: "Undecoded" }

function walkAndDiff(
  node: ParsedValue,
  path: readonly string[],
  snapshot: Map<string, FieldState>,
  events: FieldEvent[],
): void {
  const key = path.join("\0")
  let state = snapshot.get(key)

  if (!state) {
    events.push({ _tag: "field_start", path })
    state = { seenText: "", complete: false }
    snapshot.set(key, state)
  }

  if (node._tag === "object") {
    for (const [childKey, childValue] of node.entries) {
      walkAndDiff(childValue, [...path, childKey], snapshot, events)
    }
  } else if (node._tag === "array") {
    for (let index = 0; index < node.items.length; index += 1) {
      walkAndDiff(node.items[index], [...path, String(index)], snapshot, events)
    }
  } else if (node._tag === "string" || node._tag === "number") {
    if (node.value.length > state.seenText.length) {
      const delta = node.value.slice(state.seenText.length)
      events.push({ _tag: "field_delta", path, delta })
      state.seenText = node.value
    }
  }

  if (node.state === "complete" && !state.complete) {
    events.push({ _tag: "field_end", path, value: parsedValueToJson(node) })
    state.complete = true
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createStreamingFieldParser(): StreamingFieldParserErased
export function createStreamingFieldParser<A, I>(schema: Schema.Schema<A, I, never>): StreamingFieldParserConcrete<A>
export function createStreamingFieldParser(schema?: Schema.Schema.AnyNoContext): StreamingFieldParserErased {
  const jsonParser = createIncrementalJsonParser()
  const snapshot = new Map<string, FieldState>()
  const streamingSchema = schema ? deriveStreamingSchema(schema) : null
  const decodeStreaming = streamingSchema ? Schema.decodeUnknownEither(streamingSchema) : null
  let validation: ParserValidation<unknown> = { _tag: "Valid", decoded: UNDECODED }

  function markInvalid(result: ParseResult.ParseError): void {
    validation = { _tag: "Invalid", issue: formatValidationIssue(result) }
  }

  function validatePartial(): void {
    if (!decodeStreaming || validation._tag === "Invalid") return

    const partial = jsonParser.partial
    if (!partial) return

    const result = decodeStreaming(partial)
    if (result._tag === "Left") {
      markInvalid(result.left)
      return
    }

    const decoded = result.right as StreamingSchemaResult<unknown>
    validation = decoded._tag === "Complete"
      ? { _tag: "Valid", decoded: { _tag: "Decoded", value: decoded.value } }
      : { _tag: "Valid", decoded: UNDECODED }
  }

  function validateEnd(): void {
    if (!decodeStreaming || validation._tag === "Invalid") return

    validatePartial()
    if (validation._tag === "Valid" && validation.decoded._tag === "Undecoded") {
      validation = {
        _tag: "Invalid",
        issue: {
          path: jsonParser.currentPath,
          message: "Input ended before the root value completed",
        },
      }
    }
  }

  function diffPartial(): FieldEvent[] {
    const events: FieldEvent[] = []
    const partial = jsonParser.partial
    if (partial !== undefined) {
      walkAndDiff(partial, [], snapshot, events)
    }
    return events
  }

  return {
    push(chunk: string): readonly FieldEvent[] {
      jsonParser.push(chunk)
      const events = diffPartial()
      validatePartial()
      return events
    },

    end(): readonly FieldEvent[] {
      jsonParser.end()
      const events = diffPartial()
      validateEnd()
      return events
    },

    get partial(): StreamingPartial<Record<string, unknown>> | undefined {
      const p = jsonParser.partial
      if (p === undefined) return undefined
      return parsedValueToStreamingPartial(p) as StreamingPartial<Record<string, unknown>>
    },

    get decoded(): Record<string, unknown> | null {
      if (validation._tag !== "Valid" || validation.decoded._tag !== "Decoded") return null
      return validation.decoded.value as Record<string, unknown>
    },

    get valid(): boolean {
      return validation._tag === "Valid"
    },

    get validationIssue(): ValidationIssue | null {
      return validation._tag === "Invalid" ? validation.issue : null
    },
  }
}
