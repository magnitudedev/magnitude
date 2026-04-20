
import { LEAD_YIELD_TAGS } from './constants'

// =============================================================================
// Types
// =============================================================================

/**
 * Parameter binding for grammar generation.
 */
export interface GrammarParameterDef {
  readonly name: string
  readonly field: string
  readonly type: 'scalar' | 'json'
}

/**
 * Tool definition for grammar generation.
 */
export interface GrammarToolDef {
  readonly tagName: string
  readonly parameters: ReadonlyArray<GrammarParameterDef>
}

/**
 * Protocol configuration for Grammar.
 */
export interface ProtocolConfig {
  readonly minLenses: 0 | 1
  readonly allowMessages: boolean
  readonly allowTools: boolean
  readonly requiredMessageTo: string | null
  /** Maximum number of lenses allowed when a forced message is required. */
  readonly maxLenses: number | undefined
  /** Yield tags to use at turn end. */
  readonly yieldTags: ReadonlyArray<string>
  /** Available lens names. */
  readonly lensNames: ReadonlyArray<string>
  /** Tool keyword (default: "invoke"). */
  readonly toolKeyword: string
}

/**
 * Grammar configuration.
 */
export interface GrammarConfig {
  readonly tools: ReadonlyArray<GrammarToolDef>
  readonly protocol: ProtocolConfig
}

/**
 * Options for building Grammar.
 */
export interface GrammarBuildOptions {
  readonly minLenses?: 0 | 1
  readonly requiredMessageTo?: string
  readonly maxLenses?: number
  readonly yieldTags?: ReadonlyArray<string>
  readonly lensNames?: ReadonlyArray<string>
  readonly toolKeyword?: string
}

// =============================================================================
// Rule map — grammar as data, serialized at the end
// =============================================================================

type RuleMap = Map<string, string>

function serializeGrammar(rules: RuleMap): string {
  const lines: string[] = []
  for (const [name, production] of rules) {
    lines.push(`${name} ::= ${production}`)
  }
  return lines.join('\n')
}

// =============================================================================
// Defaults
// =============================================================================

const defaultProtocol: ProtocolConfig = {
  minLenses: 0,
  allowMessages: true,
  allowTools: true,
  requiredMessageTo: null,
  maxLenses: undefined,
  yieldTags: LEAD_YIELD_TAGS,
  lensNames: ['alignment', 'tasks', 'diligence', 'skills', 'turn', 'pivot'],
  toolKeyword: 'invoke',
}

// =============================================================================
// Builder
// =============================================================================

export class GrammarBuilder {
  private constructor(private readonly config: GrammarConfig) {}

  static create(tools: ReadonlyArray<GrammarToolDef>): GrammarBuilder {
    return new GrammarBuilder({
      tools,
      protocol: defaultProtocol,
    })
  }

  withMinLenses(minLenses: 0 | 1): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, minLenses },
    })
  }

  requireMessageTo(recipient: string): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, requiredMessageTo: recipient },
    })
  }

  withMaxLenses(maxLenses: number): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, maxLenses },
    })
  }

  withYieldTags(yieldTags: ReadonlyArray<string>): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, yieldTags },
    })
  }

  withLensNames(lensNames: ReadonlyArray<string>): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, lensNames },
    })
  }

  withToolKeyword(toolKeyword: string): GrammarBuilder {
    return new GrammarBuilder({
      ...this.config,
      protocol: { ...this.config.protocol, toolKeyword },
    })
  }

  withOptions(options: GrammarBuildOptions): GrammarBuilder {
    let next = this as GrammarBuilder
    if (options.minLenses !== undefined) next = next.withMinLenses(options.minLenses)
    if (options.requiredMessageTo !== undefined) next = next.requireMessageTo(options.requiredMessageTo)
    if (options.maxLenses !== undefined) next = next.withMaxLenses(options.maxLenses)
    if (options.yieldTags !== undefined) next = next.withYieldTags(options.yieldTags)
    if (options.lensNames !== undefined) next = next.withLensNames(options.lensNames)
    if (options.toolKeyword !== undefined) next = next.withToolKeyword(options.toolKeyword)
    return next
  }

  build(): string {
    const rules: RuleMap = new Map()

    this.addWhitespaceRules(rules)
    this.addLensRules(rules)
    this.addMessageRules(rules)
    this.addYieldRules(rules)
    this.addToolRules(rules)
    this.addRootRule(rules)

    return serializeGrammar(rules)
  }

  // ---------------------------------------------------------------------------
  // Rule contributors
  // ---------------------------------------------------------------------------

  private addWhitespaceRules(rules: RuleMap): void {
    // ws: unbounded whitespace between elements
    // ws1: required whitespace
    // ws-bounded: limited whitespace (1-4 chars) for capped lens slots
    rules.set('ws', '[ \\t\\n]*')
    rules.set('ws1', '[ \\t\\n]+')
    rules.set('ws-bounded', '[ \\t\\n] [ \\t\\n]? [ \\t\\n]? [ \\t\\n]?')
  }

  private addLensRules(rules: RuleMap): void {
    // think format: <|think:NAME> content <think|>
    const lensNames = this.config.protocol.lensNames
    const lensnameAlts = lensNames.map(n => `"${n}"`).join(' | ')
    
    rules.set('lensname', lensnameAlts)
    rules.set('lens', '"\\n<|think:" lensname ">\\n" think-body "\\n<think|>\\n" ws')
    rules.set('lens-tight', '"\\n<|think:" lensname ">\\n" think-body "\\n<think|>\\n" ws-bounded')

    // Generate DFA body rules for think content (tracks "<think|>" close)
    for (const rule of generateBodyRules('think', 'think')) {
      const match = rule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }
  }

  private addMessageRules(rules: RuleMap): void {
    // message format: <|message:RECIPIENT> content <message|>
    // Recipient is more open - can be user, parent, or task IDs
    rules.set('recipient', '[^ \\t\\n>]+')
    rules.set('msg', '"\\n<|message:" recipient ">\\n" msg-body "\\n<message|>\\n" ws')

    // Generate DFA body rules for message content (tracks "<message|>" close)
    for (const rule of generateBodyRules('msg', 'message')) {
      const match = rule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }

    const requiredRecipient = this.config.protocol.requiredMessageTo
    if (requiredRecipient !== null) {
      rules.set('forced-msg', `"\\n<|message:${requiredRecipient}>\\n" msg-body "\\n<message|>\\n" ws`)
    }
  }

  private addYieldRules(rules: RuleMap): void {
    // yield format: <|yield:TARGET|>
    const alternatives = this.config.protocol.yieldTags
      .map(target => `"\\n<|yield:${target}|>\\n"`)
      .join(' | ')
    rules.set('yield', alternatives)
  }

  private addToolRules(rules: RuleMap): void {
    const toolKeyword = this.config.protocol.toolKeyword
    const { toolRule, rules: toolRuleStrings } = buildToolRules(this.config.tools, toolKeyword)

    const toolMatch = toolRule.match(/^(\S+) ::= (.+)$/)
    if (toolMatch) rules.set(toolMatch[1], toolMatch[2])

    for (const rule of toolRuleStrings) {
      const match = rule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }
  }

  private addRootRule(rules: RuleMap): void {
    const parts: string[] = []
    const recipient = this.config.protocol.requiredMessageTo
    const maxLenses = this.config.protocol.maxLenses

    // Root rule structure:
    // - Lenses come first (optional, for thinking/reasoning)
    // - Then either a forced message (if required) or free-form messages/tools
    // - Finally yield to close the turn

    // Lens section
    if (recipient !== null) {
      // Forced message case: cap lenses at maxLenses to prevent infinite loops.
      const lensCount = maxLenses ?? 6
      const lensSlots: string[] = []
      for (let i = 0; i < lensCount; i++) {
        lensSlots.push('lens-tight?')
      }
      parts.push(...lensSlots)
    } else if (this.config.protocol.minLenses === 1) {
      // No forced message, but at least one lens required
      parts.push('lens+')
    } else {
      // Default: optional lenses with no constraints
      parts.push('lens*')
    }

    // Forced message (if required)
    if (recipient !== null) {
      parts.push('forced-msg')
    }

    // Middle section: messages and tools
    parts.push(this.buildMiddleAlternative())

    // Yield (turn end)
    parts.push('yield')

    rules.set('root', parts.join(' '))
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  private buildMiddleAlternative(): string {
    const middle: string[] = []
    if (this.config.protocol.allowMessages) middle.push('msg')
    if (this.config.protocol.allowTools) middle.push('invoke')
    return middle.length > 0 ? `(${middle.join(' | ')})*` : 'msg*'
  }
}

// =============================================================================
// DFA Body Rule Generation
// =============================================================================

/**
 * Generate DFA body rules that accept all 4 close tag variants:
 *   - `<tagname|>`   (canonical)
 *   - `</tagname|>`  (Mode 1: slash-prefix with pipe)
 *   - `</tagname>`   (Mode 2: slash-prefix without pipe)
 *   - `<tagname>`    (Mode 3: no slash, no pipe)
 *
 * States share tagname-tracking after the optional `/`, and accept
 * both `|>` and `>` as terminators.
 *
 * @param prefix - Rule name prefix (e.g., "think", "msg", "param")
 * @param tagName - The tag name for the close delimiter (e.g., "think", "message", "parameter")
 * @returns Array of GBNF rule strings
 */
export function generateBodyRules(prefix: string, tagName: string): string[] {
  const rules: string[] = []
  const n = tagName.length
  const s = (k: number) => `${prefix}-body-s${k}`
  const slashState = `${prefix}-body-slash`
  const pipeState = `${prefix}-body-pipe`

  rules.push(`${prefix}-body ::= ${s(0)}`)

  // s0: base state — accumulate non-'<' content, or start matching on '<'
  rules.push(`${s(0)} ::= [^<] ${s(0)} | "<" ${s(1)} | ""`)

  // s1: after '<' — accept '/' (slash variant) or first tagname char (canonical)
  const fc = escapeGbnfChar(tagName[0])
  const fcClass = escapeGbnfCharClass(tagName[0])
  rules.push(`${s(1)} ::= "/" ${slashState} | ${fc} ${s(2)} | [^</${fcClass}] ${s(0)} | ""`)

  // slashState: after '</' — expect first tagname char (shared path with canonical from s2)
  rules.push(`${slashState} ::= ${fc} ${s(2)} | [^<${fcClass}] ${s(0)} | ""`)

  // s2..s{n}: match remaining tagname chars (shared between canonical and slash paths)
  // s2 has matched tagName[0], s{k+1} has matched tagName[k]
  for (let i = 1; i < n; i++) {
    const ch = tagName[i]
    const esc = escapeGbnfChar(ch)
    const ccExclude = ch === '-' ? '[^<-]' : `[^${escapeGbnfCharClass(ch)}<]`
    rules.push(`${s(i + 1)} ::= ${esc} ${s(i + 2)} | "<" ${s(1)} | ${ccExclude} ${s(0)} | ""`)
  }

  // s{n+1}: after full tagname — accept '|' (pipe variants) or '>' (no-pipe variants)
  rules.push(`${s(n + 1)} ::= "|" ${pipeState} | ">" | "<" ${s(1)} | [^<|>] ${s(0)} | ""`)

  // pipeState: after '<[/]tagname|' — accept '>' to close
  rules.push(`${pipeState} ::= ">" | "<" ${s(1)} | [^<>] ${s(0)} | ""`)

  return rules
}

// =============================================================================
// Tool Rule Generation
// =============================================================================

export interface GrammarToolDefInternal extends GrammarToolDef {}

export function buildToolRules(
  tools: ReadonlyArray<GrammarToolDef>,
  toolKeyword: string
): { toolRule: string; rules: string[] } {
  const rules: string[] = []
  const toolNames: string[] = []

  for (const tool of tools) {
    const safeName = sanitizeRuleName(tool.tagName)
    toolNames.push(safeName)
    rules.push(...generateToolRules(safeName, tool.tagName, tool.parameters, toolKeyword))
  }

  return {
    toolRule: toolNames.length > 0 ? `invoke ::= ${toolNames.join(' | ')}` : 'invoke ::= msg',
    rules,
  }
}

export function generateToolRules(
  ruleName: string,
  tagName: string,
  parameters: ReadonlyArray<GrammarParameterDef>,
  toolKeyword: string
): string[] {
  const rules: string[] = []

  // Generate parameter rules
  const paramAlts: string[] = []
  for (const param of parameters) {
    const paramRuleName = `${ruleName}-param-${sanitizeRuleName(param.name)}`
    paramAlts.push(paramRuleName)
    
    // Parameter rule: <|parameter:name> body <parameter|>\n
    rules.push(`${paramRuleName} ::= "<|parameter:${param.name}>\\n" ${paramRuleName}-body "<parameter|>\\n" ws`)
    
    // DFA body rules for this parameter
    for (const bodyRule of generateBodyRules(`${paramRuleName}`, 'parameter')) {
      rules.push(bodyRule)
    }
  }

  // Tool invoke rule with optional filter
  // <|invoke:NAME> ws parameter* invoke-close ws
  const paramSeq = paramAlts.length > 0 ? `(${paramAlts.join(' ')})*` : ''
  
  // Invoke close: either simple <invoke|> or piped <invoke|filter>...<filter|>
  rules.push(`${ruleName} ::= "\\n<|${toolKeyword}:${tagName}>\\n" ws ${paramSeq} ${ruleName}-close ws`)
  
  // Close alternatives
  rules.push(`${ruleName}-close ::= "\\n<invoke|>\\n" | "\\n<invoke|filter>\\n" ws ${ruleName}-filter-body "\\n<filter|>\\n"`)
  
  // Filter body DFA rules
  for (const bodyRule of generateBodyRules(`${ruleName}-filter`, 'filter')) {
    rules.push(bodyRule)
  }

  return rules
}

// =============================================================================
// Utilities
// =============================================================================

export function sanitizeRuleName(tagName: string): string {
  // Remove non-alphanumeric characters and lowercase
  return `${tagName.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}tool`
}

export function escapeGbnfChar(ch: string): string {
  switch (ch) {
    case '"': return '\\"'
    case '\\': return '"\\\\"'
    case '\n': return '"\\n"'
    case '\t': return '"\\t"'
    case '<': return '"<"'
    case '>': return '">"'
    case '|': return '"|"'
    default: return `"${ch}"`
  }
}

export function escapeGbnfCharClass(ch: string): string {
  switch (ch) {
    case ']': return '\\]'
    case '\\': return '\\\\'
    case '^': return '\\^'
    case '-': return '\\-'
    default: return ch
  }
}
