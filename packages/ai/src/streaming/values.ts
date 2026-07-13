import type { JsonValue } from "../prompt/parts"
import type { ParsedValue } from "./types"

export function parsedValueToJson(node: ParsedValue): JsonValue {
  switch (node._tag) {
    case "string": return node.value
    case "number": return Number(node.value)
    case "boolean": return node.value
    case "null": return null
    case "array": return node.items.map(parsedValueToJson)
    case "object":
      return Object.fromEntries(
        node.entries.map(([key, value]) => [key, parsedValueToJson(value)]),
      )
  }
}

export function parsedValueToStreamingPartial(node: ParsedValue): unknown {
  switch (node._tag) {
    case "string":
      return node.state === "complete"
        ? { isFinal: true, value: node.value }
        : { isFinal: false, value: node.value }
    case "number":
      return node.state === "complete"
        ? { isFinal: true, value: Number(node.value) }
        : { isFinal: false, value: node.value }
    case "boolean":
      return { isFinal: true, value: node.value }
    case "null":
      return { isFinal: true, value: null }
    case "object": {
      const result: Record<string, unknown> = {}
      for (const [key, value] of node.entries) {
        result[key] = parsedValueToStreamingPartial(value)
      }
      return result
    }
    case "array":
      return node.items.map(parsedValueToStreamingPartial)
  }
}
