import type { XmlTagBinding } from './types'
import {
  validateBinding,
  type TagSchema,
  type AttributeSchema,
  type ChildTagSchema,
} from './execution/binding-validator'

export { validateBinding }
export type { TagSchema, AttributeSchema, ChildTagSchema }

export function generateBodyRules(prefix: string, tagName: string): string[] {
  const rules: string[] = []
  const closing = `</${tagName}>`
  const n = closing.length

  rules.push(`${prefix}-body ::= ${prefix}-body-s0`)

  for (let k = 0; k < n; k++) {
    const stateName = `${prefix}-body-s${k}`
    const nextStateName = `${prefix}-body-s${k + 1}`
    const ch = closing[k]

    if (k === 0) {
      rules.push(`${stateName} ::= [^<] ${stateName} | "<" ${nextStateName} | ""`)
    } else if (k < n - 1) {
      const escapedCh = escapeGbnfChar(ch)
      if (ch === '<') {
        rules.push(`${stateName} ::= "<" ${nextStateName} | [^<] ${prefix}-body-s0 | ""`)
      } else {
        const ccExcludes = ch === '-' ? `[^<-]` : `[^${escapeGbnfCharClass(ch)}<]`
        rules.push(`${stateName} ::= ${escapedCh} ${nextStateName} | "<" ${prefix}-body-s1 | ${ccExcludes} ${prefix}-body-s0 | ""`)
      }
    } else {
      rules.push(`${stateName} ::= "<" ${prefix}-body-s1 | [^<>] ${prefix}-body-s0 | ""`)
    }
  }

  return rules
}

export function sanitizeRuleName(tagName: string): string {
  return `${tagName.replace(/[^a-zA-Z0-9]/g, '')}tool`
}

export function generateToolRules(
  ruleName: string,
  tagName: string,
  schema: TagSchema,
): string[] {
  const rules: string[] = []
  const attrRuleName = `${ruleName}-attrs`

  rules.push(...generateAttrRules(attrRuleName, schema.attributes, { includeObserve: true }))

  const hasChildren = schema.children.size > 0
  const hasBody = schema.acceptsBody && !hasChildren

  if (hasChildren) {
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
    rules.push(...generateBodyRules(ruleName, tagName))
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} ">" ${ruleName}-body "</${tagName}>" ws`)
  } else {
    rules.push(`${ruleName} ::= "<${tagName}" ws ${attrRuleName} "/>" ws`)
  }

  return rules
}

export function generateAttrRules(
  attrRuleName: string,
  attributes: ReadonlyMap<string, AttributeSchema>,
  options?: { includeObserve?: boolean },
): string[] {
  const rules: string[] = []
  const attrAltName = `${attrRuleName}-alt`

  const alts: string[] = []

  if (options?.includeObserve) {
    alts.push(`"observe=\\"" ([^"] | "\\\\\\"")*  "\\""`)
  }

  for (const [attrName, attrSchema] of attributes) {
    const valuePattern = valuePatternForType(attrSchema.type)
    alts.push(`"${attrName}=\\"" ${valuePattern} "\\""`)
  }

  if (alts.length === 0) {
    rules.push(`${attrRuleName} ::= ws`)
    return rules
  }

  rules.push(`${attrAltName} ::= ${alts.join(' | ')}`)
  rules.push(`${attrRuleName} ::= (ws1 ${attrAltName})* ws`)
  return rules
}

export function generateChildRules(
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

export function valuePatternForType(type: AttributeSchema['type']): string {
  if (typeof type === 'object' && type._tag === 'enum') {
    return `(${type.values.map(v => `"${v}"`).join(' | ')})`
  }
  switch (type) {
    case 'string': return '([^"] | "\\\\\\"")*'
    case 'number': return '[0-9]+'
    case 'boolean': return '("true" | "false")'
    default: return '[^"]*'
  }
}

export function escapeGbnfChar(ch: string): string {
  switch (ch) {
    case '"': return '\\"'
    case '\\': return '"\\\\"'
    case '\n': return '"\\n"'
    case '\t': return '"\\t"'
    default: return `"${ch}"`
  }
}

export function escapeGbnfCharClass(ch: string): string {
  switch (ch) {
    case ']': return '\\]'
    case '\\': return '\\\\'
    case '^': return '\\^'
    default: return ch
  }
}

export interface GrammarToolDef {
  readonly tagName: string
  readonly binding: XmlTagBinding
  readonly inputSchema: { readonly ast: import('@effect/schema').AST.AST }
}

export function buildToolRules(tools: ReadonlyArray<GrammarToolDef>): { toolRule: string, rules: string[] } {
  const rules: string[] = []
  const toolNames: string[] = []

  for (const tool of tools) {
    const tagSchema = validateBinding(tool.tagName, tool.binding, tool.inputSchema.ast)
    const safeName = sanitizeRuleName(tool.tagName)
    toolNames.push(safeName)
    rules.push(...generateToolRules(safeName, tool.tagName, tagSchema))
  }

  return {
    toolRule: toolNames.length > 0 ? `tool ::= ${toolNames.join(' | ')}` : 'tool ::= msg',
    rules,
  }
}
