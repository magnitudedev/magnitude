
/**
 * Output Query — JSONPath-based filtering for tool output.
 * 
 * Replaces the previous XPath/fontoxpath/slimdom-based observeOutput system.
 * Uses jsonpath-plus for RFC 9535 compliant JSONPath queries.
 */

import { JSONPath } from 'jsonpath-plus'
import type { ContentPart } from '@magnitudedev/tools'

export interface QueryResult {
  /** The filtered output (subset of the full result) */
  filtered: unknown
  /** Path to the full result file for retroactive access */
  fullPath: string
  /** Whether the result was filtered (true) or is the full output (false) */
  isPartial: boolean
}

/**
 * Query a tool output using JSONPath.
 * 
 * @param output - The raw tool output object
 * @param query - JSONPath query string (e.g., '$.stdout', '$.items[0].file')
 * @param resultPath - Path to the persisted full result file
 * @returns Filtered result with metadata
 */
export function queryOutput(
  output: unknown,
  query: string | null | undefined,
  resultPath: string
): QueryResult {
  // No query or root query = full output
  if (!query || query === '.' || query === '$') {
    return {
      filtered: output,
      fullPath: resultPath,
      isPartial: false
    }
  }

  try {
    // Apply JSONPath query
    // JSONPath returns the result directly (array for most queries)
    const matches = JSONPath({ 
      path: query, 
      json: output as null | boolean | number | string | object | unknown[]
    }) as unknown[]
    
    // Unwrap single results for cleaner output
    const filtered = matches.length === 1 ? matches[0] : matches
    
    return {
      filtered,
      fullPath: resultPath,
      isPartial: true
    }
  } catch (error) {
    // On error, fall back to full output
    console.warn(`JSONPath query failed: ${query}`, error)
    return {
      filtered: output,
      fullPath: resultPath,
      isPartial: false
    }
  }
}

/**
 * Render a filtered result as ContentPart[] for LLM consumption.
 * 
 * @param toolName - Name of the tool that produced the result
 * @param filtered - The filtered output value
 * @param isPartial - Whether this is a partial (filtered) result
 * @param fullPath - Path to the full result file
 * @returns Array of content parts for the LLM context
 */
export function renderFilteredResult(
  toolName: string,
  filtered: unknown,
  isPartial: boolean,
  fullPath: string
): ContentPart[] {
  const parts: ContentPart[] = []
  
  // Build the result block
  const resultText = renderResultBlock(toolName, filtered)
  
  // Add the result
  parts.push({ type: 'text', text: resultText })
  
  // If partial, add the full result reference
  if (isPartial) {
    parts.push({ type: 'text', text: `\nFull result: ${fullPath}` })
  }
  
  return parts
}

/**
 * Render a result block using result/out syntax.
 * 
 * Rules from spec §8.3:
 * - Pure string outputs: content is the entire result block body (no out wrapper)
 * - Void outputs: empty result block
 * - Object outputs: <|out:fieldName>value<out|> for each field
 * - Top-level string fields: raw text (no escaping)
 * - Complex types: JSON representation
 */
export function renderResultBlock(toolName: string, output: unknown): string {
  // Void/null/undefined
  if (output === undefined || output === null) {
    return `<|result:${toolName}>\n<result|>`
  }
  
  // Pure string (like read output)
  if (typeof output === 'string') {
    return `<|result:${toolName}>\n${output}\n<result|>`
  }
  
  // Scalar primitives (number, boolean)
  if (typeof output === 'number' || typeof output === 'boolean') {
    return `<|result:${toolName}>\n${String(output)}\n<result|>`
  }
  
  // Arrays - render as JSON within out block
  if (Array.isArray(output)) {
    const json = JSON.stringify(output)
    return `<|result:${toolName}>\n<|out:items>${json}<out|>\n<result|>`
  }
  
  // Objects - render each field
  if (typeof output === 'object' && output !== null) {
    const lines: string[] = [`<|result:${toolName}>`]
    
    for (const [key, value] of Object.entries(output)) {
      const rendered = renderOutField(key, value)
      lines.push(rendered)
    }
    
    lines.push(`<result|>`)
    return lines.join('\n')
  }
  
  // Fallback
  return `<|result:${toolName}>\n${String(output)}\n<result|>`
}

/**
 * Render a single out field.
 * 
 * Rules:
 * - String values: raw text, multi-line if contains newlines
 * - Other values: JSON inline
 */
function renderOutField(name: string, value: unknown): string {
  // String values - raw text
  if (typeof value === 'string') {
    if (value.includes('\n') || value.length > 80) {
      // Multi-line or long string
      return `<|out:${name}>\n${value}\n<out|>`
    } else {
      // Short inline string
      return `<|out:${name}>${value}<out|>`
    }
  }
  
  // Other values - JSON
  const json = JSON.stringify(value)
  return `<|out:${name}>${json}<out|>`
}

/**
 * Legacy adapter: Convert a result to ContentPart[] using the new rendering.
 * 
 * This replaces the old observeOutput function signature for compatibility
 * during migration, but uses the new JSONPath-based system internally.
 */
export function observeOutput(
  output: unknown,
  query: string | undefined,
  toolName: string,
  turnId: string,
  callId: string,
  resultPath: string
): ContentPart[] {
  const { filtered, isPartial } = queryOutput(output, query, resultPath)
  return renderFilteredResult(toolName, filtered, isPartial, resultPath)
}

/**
 * Common JSONPath query patterns for convenience.
 */
export const QueryPatterns = {
  /** Full output */
  full: '$',
  /** First item in an array */
  first: '$[0]',
  /** All items */
  all: '$[*]',
  /** Count of items */
  count: '$.length',
  /** Specific field */
  field: (name: string) => `$.${name}`,
  /** Nested field */
  nested: (...path: string[]) => `$.${path.join('.')}`,
  /** Array element by index */
  index: (idx: number) => `$[${idx}]`,
  /** Filter by condition */
  filter: (condition: string) => `$[?${condition}]`,
} as const
