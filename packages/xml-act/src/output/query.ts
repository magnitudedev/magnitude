/**
 * Output Query — JSONPath-based filtering for tool output.
 * 
 * Replaces the previous XPath/fontoxpath/slimdom-based observeOutput system.
 * Uses jsonpath-plus for RFC 9535 compliant JSONPath queries.
 */

import { JSONPath } from 'jsonpath-plus'
import type { ContentPart } from '@magnitudedev/tools'
import { renderResult, renderResultToParts } from './renderer'

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
  // Delegate to renderer (handles image detection automatically)
  const parts = renderResultToParts(toolName, filtered)
  
  // If partial, add the full result reference
  if (isPartial) {
    parts.push({ type: 'text', text: `\nFull result: ${fullPath}` })
  }
  
  return parts
}
