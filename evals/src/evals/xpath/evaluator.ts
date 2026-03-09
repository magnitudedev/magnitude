/**
 * XPath 3.1 query evaluator
 *
 * Runs XPath expressions against XML documents and/or JSON variables
 * using fontoxpath + slimdom, then compares results to expected values.
 */

import { evaluateXPath } from 'fontoxpath'
import { parseXmlDocument } from 'slimdom'
import type { XPathScenario } from './scenarios'

export interface EvalResult {
  /** Whether the query produced the expected result */
  match: boolean
  /** The actual value(s) produced by the query */
  actual: unknown
  /** Error message if the query failed to parse/execute */
  error?: string
}

/**
 * Evaluate an XPath 3.1 expression against the scenario's data
 * and compare the result to the expected value.
 */
export function evaluateQuery(query: string, scenario: XPathScenario): EvalResult {
  try {
    const doc = scenario.xml ? parseXmlDocument(scenario.xml) : null
    const variables = scenario.variables ?? {}

    const returnType = scenario.isSequence
      ? evaluateXPath.ALL_RESULTS_TYPE
      : evaluateXPath.ANY_TYPE

    const raw = evaluateXPath(query, doc, null, variables, returnType)
    const actual = unwrapResult(raw)

    const match = deepEqual(actual, scenario.expected)

    return { match, actual }
  } catch (e) {
    return {
      match: false,
      actual: undefined,
      error: e instanceof Error ? e.message : String(e),
    }
  }
}

/**
 * Unwrap XPath result values. DOM nodes (attributes, text nodes, elements)
 * get converted to their string value. Arrays are recursively unwrapped.
 */
function unwrapResult(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(unwrapResult)
  }
  // DOM Node — has nodeType property
  if (value !== null && typeof value === 'object' && 'nodeType' in value) {
    const node = value as { nodeType: number; value?: string; textContent?: string | null }
    // Attribute node (nodeType 2) — use .value
    if (node.nodeType === 2) return node.value ?? ''
    // Text node (nodeType 3) or element — use .textContent
    return node.textContent ?? ''
  }
  return value
}

/**
 * Deep equality comparison that handles:
 * - primitives (string, number, boolean)
 * - arrays of primitives (order-sensitive)
 * - type coercion for numbers that come back as strings from XPath
 */
function deepEqual(actual: unknown, expected: unknown): boolean {
  // Both null/undefined
  if (actual == null && expected == null) return true
  if (actual == null || expected == null) return false

  // Array comparison
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual)) return false
    if (actual.length !== expected.length) return false
    return expected.every((exp, i) => valueEqual(actual[i], exp))
  }

  return valueEqual(actual, expected)
}

/**
 * Compare two scalar values, allowing numeric string coercion.
 * XPath may return "5" when we expect 5, or 92.1 when we expect "92.1".
 */
function valueEqual(actual: unknown, expected: unknown): boolean {
  if (actual === expected) return true

  // Numeric coercion: "5" == 5, 92.1 == "92.1"
  if (typeof expected === 'number' && typeof actual === 'string') {
    return Number(actual) === expected
  }
  if (typeof expected === 'string' && typeof actual === 'number') {
    return actual === Number(expected)
  }
  // String comparison with toString
  if (typeof expected === 'string' && typeof actual === 'string') {
    return actual.trim() === expected.trim()
  }

  return false
}
