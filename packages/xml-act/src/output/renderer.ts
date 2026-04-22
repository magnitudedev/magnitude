/**
 * Output Renderer — Renders tool output using result/out syntax.
 * 
 * This module provides the rendering layer for tool results in the format,
 * replacing the previous XML-based output serialization.
 * 
 * Design principles from spec §8:
 * - Mirror the input syntax (<|result:NAME> like <|invoke:NAME>)
 * - Top-level string fields render as raw text (no escaping)
 * - Complex types render as JSON
 * - Pure string outputs have no out wrapper
 * - Void outputs are empty result blocks
 */

import type { ContentPart, ImageMediaType } from '@magnitudedev/tools'

/**
 * Configuration for result rendering.
 */
export interface RenderConfig {
  /** Whether to include the full result path in partial results */
  includeFullPath?: boolean
  /** Full result file path (for partial results) */
  fullPath?: string
}

/**
 * Shape of an image output produced by a tool.
 */
interface ImageOutput {
  base64: string
  mediaType: ImageMediaType
  width: number
  height: number
}

/**
 * Check if a tool output is an image output.
 */
function isImageOutput(output: unknown): output is ImageOutput {
  if (typeof output !== 'object' || output === null) return false
  const o = output as Record<string, unknown>
  return (
    typeof o.base64 === 'string' &&
    typeof o.mediaType === 'string' &&
    o.mediaType.startsWith('image/') &&
    typeof o.width === 'number' &&
    typeof o.height === 'number'
  )
}

/**
 * Render a tool result to a formatted string.
 * 
 * This is the main entry point for result rendering.
 * 
 * @param toolName - Name of the tool (e.g., 'read', 'shell', 'grep')
 * @param output - The tool output value
 * @param config - Optional rendering configuration
 * @returns Formatted result string
 */
export function renderResult(
  toolName: string,
  output: unknown,
  config?: RenderConfig
): string {
  const resultBody = renderResultBody(toolName, output)
  
  // Add full path reference if this is a partial result
  if (config?.includeFullPath && config.fullPath) {
    return `${resultBody}\nFull result: ${config.fullPath}`
  }
  
  return resultBody
}

/**
 * Render just the result block body (without the full path reference).
 */
export function renderResultBody(toolName: string, output: unknown): string {
  // Void/null/undefined outputs (like write)
  if (output === undefined || output === null) {
    return renderVoidResult(toolName)
  }
  
  // Pure string outputs (like read)
  if (typeof output === 'string') {
    return renderStringResult(toolName, output)
  }
  
  // Number/boolean outputs
  if (typeof output === 'number' || typeof output === 'boolean') {
    return renderScalarResult(toolName, output)
  }
  
  // Array outputs (like grep, tree)
  if (Array.isArray(output)) {
    return renderArrayResult(toolName, output)
  }
  
  // Object outputs (like shell with {stdout, stderr, exitCode})
  if (typeof output === 'object') {
    return renderObjectResult(toolName, output as Record<string, unknown>)
  }
  
  // Fallback
  return renderScalarResult(toolName, output)
}

/**
 * Render a void result (no content).
 */
export function renderVoidResult(toolName: string): string {
  return `<result tool="${toolName}"/>`
}

/**
 * Render a pure string result (content is the entire block body).
 */
export function renderStringResult(toolName: string, content: string): string {
  return `<result tool="${toolName}">\n${content}\n</result>`
}

/**
 * Render a scalar (number/boolean) result.
 */
export function renderScalarResult(toolName: string, value: number | boolean | unknown): string {
  return `<result tool="${toolName}">\n${String(value)}\n</result>`
}

/**
 * Render an array result as JSON within an out block.
 */
export function renderArrayResult(toolName: string, items: unknown[]): string {
  const json = JSON.stringify(items)
  return `<result tool="${toolName}">\n<out name="items">${json}</out>\n</result>`
}

/**
 * Render an object result with out fields.
 */
export function renderObjectResult(
  toolName: string,
  fields: Record<string, unknown>
): string {
  const lines: string[] = [`<result tool="${toolName}">`]
  
  for (const [name, value] of Object.entries(fields)) {
    lines.push(renderOutField(name, value))
  }
  
  lines.push(`</result>`)
  return lines.join('\n')
}

/**
 * Render a single out field.
 * 
 * Rules from spec §8.3:
 * - String fields: raw text, multi-line if contains newlines or is long
 * - Non-string fields: JSON inline
 */
export function renderOutField(name: string, value: unknown): string {
  // String values - raw text (no escaping)
  if (typeof value === 'string') {
    // Multi-line or long strings get their own lines
    if (value.includes('\n') || value.length > 80) {
      return `<out name="${name}">\n${value}\n</out>`
    }
    // Short strings inline
    return `<out name="${name}">${value}</out>`
  }
  
  // Non-string values - JSON
  const json = JSON.stringify(value)
  return `<out name="${name}">${json}</out>`
}

/**
 * Render a result to ContentPart[] for LLM context injection.
 * 
 * If the output is an image, returns an image ContentPart instead of text.
 */
export function renderResultToParts(
  toolName: string,
  output: unknown,
  config?: RenderConfig
): ContentPart[] {
  if (isImageOutput(output)) {
    return [{
      type: 'image',
      base64: output.base64,
      mediaType: output.mediaType,
      width: output.width,
      height: output.height,
    }]
  }
  const text = renderResult(toolName, output, config)
  return [{ type: 'text', text }]
}

/**
 * Type-specific renderers for common tool output patterns.
 */

/**
 * Render shell command output.
 */
export function renderShellResult(
  stdout: string,
  stderr: string,
  exitCode: number,
  mode: 'completed' | 'timeout' | 'error' = 'completed',
  config?: RenderConfig
): string {
  const output: Record<string, unknown> = {
    mode,
    exitCode,
    stdout,
  }
  // Only include stderr if it's non-empty
  if (stderr && stderr.trim()) {
    output.stderr = stderr
  }
  return renderResult('shell', output, config)
}

/**
 * Render grep results.
 */
export function renderGrepResult(
  items: Array<{ file: string; match: string }>,
  config?: RenderConfig
): string {
  return renderResult('grep', items, config)
}

/**
 * Render read file result.
 */
export function renderReadResult(content: string, config?: RenderConfig): string {
  return renderResult('read', content, config)
}

/**
 * Render write file result (void).
 */
export function renderWriteResult(config?: RenderConfig): string {
  return renderResult('write', null, config)
}

/**
 * Render edit file result.
 */
export function renderEditResult(summary: string, config?: RenderConfig): string {
  return renderResult('edit', summary, config)
}

/**
 * Render tree listing result.
 */
export function renderTreeResult(
  entries: Array<{ path: string; name: string; type: string; depth: number }>,
  config?: RenderConfig
): string {
  return renderResult('tree', entries, config)
}

/**
 * Render skill result.
 */
export function renderSkillResult(
  content: string,
  skillPath: string,
  config?: RenderConfig
): string {
  const output = {
    skillPath,
    content,
  }
  return renderResult('skill', output, config)
}

/**
 * Parse a result block (for testing/validation).
 * This is a simple parser for the result format.
 */
export function parseResultBlock(text: string): {
  toolName: string
  fields: Map<string, unknown>
  content?: string
} | null {
  const trimmed = text.trim()

  // Self-closing void result: <result tool="NAME"/>
  const voidMatch = trimmed.match(/^<result tool="([^"]+)"\/>\s*$/)
  if (voidMatch) {
    return { toolName: voidMatch[1], fields: new Map() }
  }

  // Open tag: <result tool="NAME">
  const openMatch = trimmed.match(/^<result tool="([^"]+)">\n?/)
  if (!openMatch) return null

  const toolName = openMatch[1]
  const remaining = trimmed.slice(openMatch[0].length)

  // Check for pure string result (no out blocks)
  const closeMatch = remaining.match(/\n?<\/result>\s*$/)
  if (closeMatch) {
    const content = remaining.slice(0, -closeMatch[0].length)

    // Check if it contains out blocks
    if (!content.includes('<out name=')) {
      return { toolName, fields: new Map(), content }
    }
  }

  // Parse out blocks: <out name="NAME">VALUE</out>
  const fields = new Map<string, unknown>()
  const outRegex = /<out name="([^"]+)">([\s\S]*?)<\/out>/g
  let match

  while ((match = outRegex.exec(remaining)) !== null) {
    const name = match[1]
    const value = match[2].trim()

    // Try to parse as JSON, fall back to string
    try {
      fields.set(name, JSON.parse(value))
    } catch {
      fields.set(name, value)
    }
  }

  return { toolName, fields }
}

/**
 * Validation utilities.
 */

/**
 * Check if a string is a valid result block.
 */
export function isValidResultBlock(text: string): boolean {
  const trimmed = text.trim()
  return (
    /^<result tool="[^"]+"\/>$/.test(trimmed) ||
    /^<result tool="[^"]+">[\s\S]*<\/result>$/.test(trimmed)
  )
}

/**
 * Extract the tool name from a result block.
 */
export function extractToolName(text: string): string | null {
  const match = text.match(/^<result tool="([^"]+)"/)
  return match ? match[1] : null
}
