/**
 * Input Builder — builds tool input from parsed parameters.
 *
 * Uses derived ParameterSchema to map flat parameter values to the tool's
 * expected input shape. No manual binding needed — parameter names ARE
 * field paths (with dots for nested fields).
 */

import type { ParameterSchema, ScalarType } from './parameter-schema'

// =============================================================================
// Coercion Logic
// =============================================================================

const TRUTHY = new Set(['true', 'True', 'TRUE', '1'])
const FALSY = new Set(['false', 'False', 'FALSE', '0'])

/**
 * Coerce a raw string value to the expected scalar type.
 */
export function coerceParameterValue(raw: string, type: ScalarType): string | number | boolean {
  if (typeof type === 'object' && type._tag === 'enum') {
    if (!type.values.includes(raw)) {
      throw new Error(`Invalid enum value '${raw}' — expected: ${type.values.join(', ')}`)
    }
    return raw
  }

  switch (type) {
    case 'string':
      return raw
    case 'number': {
      if (raw === '' || raw === 'NaN' || raw === 'Infinity' || raw === '-Infinity') {
        throw new Error(`Cannot coerce '${raw}' to number`)
      }
      const n = Number(raw)
      if (isNaN(n)) throw new Error(`Cannot coerce '${raw}' to number`)
      return n
    }
    case 'boolean': {
      if (TRUTHY.has(raw)) return true
      if (FALSY.has(raw)) return false
      throw new Error(`Cannot coerce '${raw}' to boolean — expected: true, false, True, False, TRUE, FALSE, 0, 1`)
    }
    default:
      return raw
  }
}

// =============================================================================
// Nested Value Setting
// =============================================================================

/** Set a value at a dotted path, creating intermediate objects as needed. */
function setNestedValue(obj: Record<string, unknown>, path: string, value: unknown): void {
  const segments = path.split('.')
  let current = obj
  for (let i = 0; i < segments.length - 1; i++) {
    if (!(segments[i] in current)) {
      current[segments[i]] = {}
    }
    current = current[segments[i]] as Record<string, unknown>
  }
  current[segments[segments.length - 1]] = value
}

// =============================================================================
// Public API
// =============================================================================

export interface ParsedParameter {
  name: string
  value: string
  isComplete: boolean
}

export interface ParsedInvoke {
  readonly tagName: string
  readonly toolCallId: string
  readonly parameters: ReadonlyMap<string, ParsedParameter>
  readonly filter?: string
}

/**
 * Build a tool input object from parsed parameters and their derived schemas.
 *
 * Parameter names ARE field paths — e.g., parameter 'options.type' sets
 * input.options.type. No binding mapping needed.
 *
 * @param parsed - The parsed invoke with parameter values
 * @param parameterSchemas - Map of parameter names to their schemas (from deriveParameters)
 * @returns The constructed input object with properly coerced values
 * @throws If a required parameter is missing, JSON parsing fails, or coercion fails
 */
export function buildInput(
  parsed: ParsedInvoke,
  parameterSchemas: ReadonlyMap<string, ParameterSchema>,
): Record<string, unknown> {
  const input: Record<string, unknown> = {}

  for (const [paramName, schema] of parameterSchemas) {
    const parsedParam = parsed.parameters.get(paramName)

    if (!parsedParam) {
      if (schema.required) {
        throw new Error(`Required parameter '${paramName}' is missing`)
      }
      // Optional parameter not provided — skip, don't set
      continue
    }

    if (!parsedParam.isComplete) {
      throw new Error(`Parameter '${paramName}' is incomplete`)
    }

    if (schema.type === 'json') {
      // JSON parameter — parse the JSON value
      try {
        const parsedJson = JSON.parse(parsedParam.value)
        setNestedValue(input, paramName, parsedJson)
      } catch (e) {
        throw new Error(`Invalid JSON for parameter '${paramName}': ${e}`)
      }
    } else {
      // Scalar parameter — coerce to the correct type
      const coerced = coerceParameterValue(parsedParam.value, schema.type)
      setNestedValue(input, paramName, coerced)
    }
  }

  return input
}
