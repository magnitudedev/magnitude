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

  // Lens - uses DFA body to allow any content including angle brackets
  rules.push('lens ::= "<lens name=\\"" lensname "\\">" lens-body "</lens>" ws')
  rules.push('lensname ::= "alignment" | "tasks" | "diligence" | "skills" | "turn" | "pivot"')
  rules.push(...generateBodyRules('lens', 'lens'))

  // Message - uses DFA body to allow any content including angle brackets
  rules.push('msg ::= "<message to=\\"" [^"]* "\\">" msg-body "</message>" ws')
  rules.push(...generateBodyRules('msg', 'message'))

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
// DFA body rule generation
// =============================================================================

/**
 * Generate GBNF rules for a body that can contain any content EXCEPT
 * the exact closing tag string `</tagName>`.
 *
 * Uses a DFA (deterministic finite automaton) encoded as recursive GBNF rules.
 * Each state tracks how many characters of `</tagName>` have been matched so far.
 *
 * State 0: no partial match in progress
 * State 1: seen `<`
 * State 2: seen `</`
 * State 3: seen `</t` (first char of tagName)
 * ...
 * State N+2: seen `</tagName` (all chars of tagName matched)
 *   - if next char is `>`, the closing tag is complete → body ends (no alternative)
 *   - otherwise, reset
 *
 * @param prefix  Rule name prefix (e.g. "writetool")
 * @param tagName The XML tag name (e.g. "write") used to form the closing tag
 * @returns Array of GBNF rule strings
 */
export function generateBodyRules(prefix: string, tagName: string): string[] {
  const rules: string[] = []
  const closing = `</${tagName}>`
  // closing sequence characters: '<', '/', ...tagName chars..., '>'
  // States: s0 = normal, s1 = seen '<', s2 = seen '</', s(2+i) = seen closing[0..1+i]
  // Total states: 2 + tagName.length + 1 = tagName.length + 3
  // But we only need states for partial matches of the closing sequence.
  // closing = ['<', '/', t, a, g, N, a, m, e, '>']
  // State k means we've matched closing[0..k-1] (k chars of closing)
  const n = closing.length // total chars in closing tag string

  // Entry rule: just references s0
  rules.push(`${prefix}-body ::= ${prefix}-body-s0`)

  for (let k = 0; k < n; k++) {
    const stateName = `${prefix}-body-s${k}`
    const nextStateName = `${prefix}-body-s${k + 1}`
    const ch = closing[k]

    if (k === 0) {
      // State 0: normal. 
      // - any char that isn't `<` → stay in s0
      // - `<` → go to s1 (we've matched `<`)
      // - empty → body ends
      rules.push(`${stateName} ::= [^<] ${stateName} | "<" ${nextStateName} | ""`)
    } else if (k < n - 1) {
      // Intermediate state: we've matched closing[0..k-1], next expected is closing[k]
      // - if next char == closing[k] → advance to s(k+1)
      // - if next char == '<' → go to s1 (restart partial match from '<')
      // - if next char is anything else → go back to s0
      // Special case: closing[k] might itself be '<' (it's not for normal tag names, but handle it)
      const escapedCh = escapeGbnfChar(ch)
      if (ch === '<') {
        // Expected char is '<': matching it advances state; no separate '<' branch needed
        rules.push(`${stateName} ::= "<" ${nextStateName} | [^<] ${prefix}-body-s0 | ""`)
      } else {
        const ccExcludes = ch === '-' ? `[^<-]` : `[^${escapeGbnfCharClass(ch)}<]`
        rules.push(`${stateName} ::= ${escapedCh} ${nextStateName} | "<" ${prefix}-body-s1 | ${ccExcludes} ${prefix}-body-s0 | ""`)
      }
    } else {
      // Final state: we've matched closing[0..n-2] = all of `</tagName`
      // Next char would be `>` which completes the closing tag → body MUST end here
      // Any other char: go back to appropriate state
      // closing[n-1] is '>'
      // - if next char == '>' → DO NOT consume, body ends (empty alternative)
      // - if next char == '<' → go to s1
      // - otherwise → go back to s0
      rules.push(`${stateName} ::= "<" ${prefix}-body-s1 | [^<>] ${prefix}-body-s0 | ""`)
    }
  }

  return rules
}

function escapeGbnfChar(ch: string): string {
  // Returns a GBNF literal for a single character
  switch (ch) {
    case '"': return '\\"'
    case '\\': return '"\\\\"'
    case '\n': return '"\\n"'
    case '\t': return '"\\t"'
    default: return `"${ch}"`
  }
}

function escapeGbnfCharClass(ch: string): string {
  // Returns a character suitable for use inside [...] character class
  // Note: '-' is handled specially at the call site by placing it at the end
  switch (ch) {
    case ']': return '\\]'
    case '\\': return '\\\\'
    case '^': return '\\^'
    default: return ch
  }
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
    // Tool with body content — use DFA body rules
    rules.push(...generateBodyRules(ruleName, tagName))
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} ">" ${ruleName}-body "</${tagName}>" ws`)
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
    rules.push(...generateBodyRules(ruleName, tagName))
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} ">" ${ruleName}-body "</${tagName}>" ws`)
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
