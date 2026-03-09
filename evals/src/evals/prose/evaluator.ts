/**
 * Prose delimiter evaluator
 *
 * Check functions that inspect resolved tool arguments from the test sandbox
 * to verify correct prose delimiter usage.
 */

import type { CheckResult } from '../../types'
import type { CapturedCall, TestSandboxResult } from '../../test-sandbox'
import { findCalls } from '../../test-sandbox'

// =============================================================================
// Check helpers — inspect resolved string values from sandbox execution
// =============================================================================

/**
 * Check if a resolved string value contains unwanted backslash escaping.
 * These are strings that have already been through the full sandbox pipeline,
 * so any backslash-backtick or backslash-dollar is a genuine escaping error.
 */
export function checkNoUnwantedEscaping(value: unknown, context: string): CheckResult {
  if (typeof value !== 'string') return { passed: true }

  // Check for escaped backticks: literal \` in the resolved string
  if (value.includes('\\`')) {
    const idx = value.indexOf('\\`')
    return {
      passed: false,
      message: 'Found escaped backtick (\\`) in ' + context,
      snippet: value.slice(Math.max(0, idx - 30), Math.min(value.length, idx + 30))
    }
  }

  // Check for escaped dollar signs: literal \$ in the resolved string
  if (value.includes('\\$')) {
    const idx = value.indexOf('\\$')
    return {
      passed: false,
      message: 'Found escaped dollar sign (\\$) in ' + context,
      snippet: value.slice(Math.max(0, idx - 30), Math.min(value.length, idx + 30))
    }
  }

  return { passed: true }
}

/**
 * Check that a resolved string uses real newlines, not literal \\n sequences.
 * 
 * In the resolved sandbox output, a real newline is \n (char 10).
 * A literal backslash-n means the model wrote \\n inside prose delimiters,
 * which the sandbox preserved as literal characters (not a newline).
 * 
 * Only flags when the string ALSO contains real newlines — indicating the model
 * mixed real and escaped newlines inconsistently.
 */
export function checkNoLiteralNewlineEscapes(value: unknown, context: string): CheckResult {
  if (typeof value !== 'string') return { passed: true }
  if (!value.includes('\n')) return { passed: true } // No real newlines, can't judge

  // Look for literal backslash followed by 'n' in the resolved string
  const idx = value.indexOf('\\n')
  if (idx !== -1) {
    // Skip if it's inside a JS string literal in code content (e.g. .join("\\n"))
    const before = value.slice(Math.max(0, idx - 1), idx)
    const after = value.slice(idx + 2, idx + 3)
    if ((before === '"' || before === "'") && (after === '"' || after === "'")) {
      return { passed: true }
    }

    return {
      passed: false,
      message: 'Found literal \\n in ' + context + ' (should use real newline)',
      snippet: value.slice(Math.max(0, idx - 30), Math.min(value.length, idx + 30))
    }
  }

  return { passed: true }
}
