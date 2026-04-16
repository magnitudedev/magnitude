/**
 * Dynamic GBNF Grammar Generator
 *
 * Generates GBNF grammar from XML tool bindings for use with
 * constrained generation APIs (e.g. Fireworks AI).
 *
 * Takes tool definitions and produces a complete GBNF grammar string
 * that constrains model output to valid XML-act protocol with per-tool
 * attribute and body/child validation.
 */

import { AST } from '@effect/schema'
import type { XmlTagBinding } from './types'
import { validateBinding, type TagSchema, type AttributeSchema, type ChildTagSchema } from './execution/binding-validator'

// =============================================================================
// Public API
// =============================================================================

export interface GrammarToolDef {
  readonly tagName: string
  readonly binding: XmlTagBinding
  readonly inputSchema: { readonly ast: AST.AST }
}

/**
 * Generate a complete GBNF grammar from tool definitions.
 *
 * Produces a grammar that constrains output to the XML-act protocol:
 * - Lenses, messages, and tools in any order
 * - Per-tool rules for valid tag names, attributes, body/children
 * - End-turn control
 */
export function generateGrammar(tools: ReadonlyArray<GrammarToolDef>): string {
  const rules: string[] = []

  // Whitespace
  rules.push('ws ::= [ \\t\\n]*')

  // Root
  rules.push('root ::= lens* (msg | tool)* endturn')

  // Lens
  rules.push('lens ::= "<lens name=\\"" lensname "\\">" [^<]+ "</lens>" ws')
  rules.push('lensname ::= "alignment" | "tasks" | "diligence" | "skills" | "turn" | "pivot"')

  // Message
  rules.push('msg ::= "<message to=\\"" [^"]* "\\">" [^<]+ "</message>" ws')

  // End-turn
  rules.push('endturn ::= "<end-turn>" ws ("<idle/>" | "<continue/>") ws "</end-turn>"')

  // Build per-tool rules and collect tool alternation
  const toolNames: string[] = []
  for (const tool of tools) {
    const tagSchema = validateBinding(tool.tagName, tool.binding, tool.inputSchema.ast)
    const safeName = sanitizeRuleName(tool.tagName)
    toolNames.push(safeName)
    rules.push(...generateToolRules(safeName, tool.tagName, tagSchema))
  }

  // Tool alternation
  if (toolNames.length > 0) {
    rules.unshift(`tool ::= ${toolNames.join(' | ')}`)
  } else {
    rules.unshift('tool ::= msg')  // fallback: no tools, match message
  }

  return rules.join('\n')
}

// =============================================================================
// Tool rule generation
// =============================================================================

function sanitizeRuleName(tagName: string): string {
  // GBNF rule names can't contain slashes, dots, etc.
  return `${tagName.replace(/[^a-zA-Z0-9]/g, '')}tool`
}

function generateToolRules(
  ruleName: string,
  tagName: string,
  schema: TagSchema,
): string[] {
  const rules: string[] = []
  const attrRuleName = `${ruleName}-attrs`

  // Generate attribute rules (include observe for top-level tool attrs)
  rules.push(...generateAttrRules(attrRuleName, schema.attributes, { includeObserve: true }))

  // Determine tool shape
  const hasChildren = schema.children.size > 0
  const hasBody = schema.acceptsBody && !hasChildren

  if (hasChildren) {
    // Tool with child elements
    const childAltParts: string[] = []
    for (const [childTag, childSchema] of schema.children) {
      const childRuleName = `${ruleName}-${sanitizeRuleName(childTag)}`
      rules.push(...generateChildRules(childRuleName, childTag, childSchema))
      childAltParts.push(childRuleName)
    }
    const childAltName = `${ruleName}-children`
    rules.push(`${childAltName} ::= (${childAltParts.join(' | ')})*`)

    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} ">" ws ${childAltName} ws "</${tagName}>" ws`)
  } else if (hasBody) {
    // Tool with body content
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} ">" [^<]+ "</${tagName}>" ws`)
  } else {
    // Self-closing tool
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} "/>" ws`)
  }

  return rules
}

function generateAttrRules(
  attrRuleName: string,
  attributes: ReadonlyMap<string, AttributeSchema>,
  options?: { includeObserve?: boolean },
): string[] {
  const rules: string[] = []
  const attrAltName = `${attrRuleName}-alt`

  // Collect all attribute alternatives
  const alts: string[] = []

  // Add observe attribute if requested (framework metadata, not part of tool bindings)
  if (options?.includeObserve) {
    alts.push(`"observe=\\"" [^"]* "\\""`)
  }

  for (const [attrName, attrSchema] of attributes) {
    const valuePattern = valuePatternForType(attrSchema.type)
    alts.push(`"${attrName}=\\"" ${valuePattern} "\\""`)
  }

  if (alts.length === 0) {
    rules.push(`${attrRuleName} ::= ws`)
    return rules
  }

  // Order-independent: (ws (alt1 | alt2 | ...))* ws
  // Trailing ws allows whitespace before closing > or />
  rules.push(`${attrAltName} ::= ${alts.join(' | ')}`)
  rules.push(`${attrRuleName} ::= (ws ${attrAltName})* ws`)
  return rules
}

function generateChildRules(
  ruleName: string,
  tagName: string,
  schema: ChildTagSchema,
): string[] {
  const rules: string[] = []

  const attrRuleName = `${ruleName}-attrs`
  rules.push(...generateAttrRules(attrRuleName, schema.attributes))

  if (schema.acceptsBody) {
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} ">" [^<]+ "</${tagName}>" ws`)
  } else {
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} "/>" ws`)
  }

  return rules
}

function valuePatternForType(type: 'string' | 'number' | 'boolean'): string {
  switch (type) {
    case 'string': return '[^"]*'
    case 'number': return '[0-9]+'
    case 'boolean': return '("true" | "false")'
  }
}
