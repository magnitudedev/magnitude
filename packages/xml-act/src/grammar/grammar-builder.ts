
import { LEAD_YIELD_TAGS } from '../constants'

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
    this.addSharedBodyRules(rules)
    this.addLensRules(rules)
    this.addMessageRules(rules)
    this.addYieldRules(rules)
    this.addInvokeCloseRule(rules)
    this.addToolRules(rules)
    this.addRootRule(rules)

    return serializeGrammar(rules)
  }

  // ---------------------------------------------------------------------------
  // Rule contributors
  // ---------------------------------------------------------------------------

  private addWhitespaceRules(rules: RuleMap): void {
    // ws: unbounded whitespace (spaces, tabs, newlines) — used before block elements where model naturally produces whitespace
    rules.set('ws', '[ \\t\\n]*')
    // bhws: bounded horizontal whitespace (0-4 chars, spaces/tabs only) — safe inline spacing that cannot trap the model
    rules.set('bhws', '[ \\t]? [ \\t]? [ \\t]? [ \\t]?')
  }

  private addSharedBodyRules(rules: RuleMap): void {
    // Shared DFAs — one per close tag name. No "" exits; body must end with close tag.
    for (const bodyRule of generateBodyRules('param', 'parameter')) {
      const match = bodyRule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }
    for (const bodyRule of generateBodyRules('think', 'think')) {
      const match = bodyRule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }
    for (const bodyRule of generateBodyRules('msg', 'message')) {
      const match = bodyRule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }
    for (const bodyRule of generateBodyRules('filter', 'filter')) {
      const match = bodyRule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }
  }

  private addLensRules(rules: RuleMap): void {
    const lensNames = this.config.protocol.lensNames
    const lensnameAlts = lensNames.map(n => `"${n}"`).join(' | ')
    
    rules.set('lensname', lensnameAlts)
    // DFA handles close tag — no literal "<think|>" in outer rule
    rules.set('lens', 'ws "<|think:" lensname ">" "\\n" think-body')
    rules.set('lens-cap', 'ws "<|think:" lensname ">" "\\n" think-body')
  }

  private addMessageRules(rules: RuleMap): void {
    rules.set('recipient', '[^ \\t\\n>]+')
    // DFA handles close tag — no literal "<message|>" in outer rule
    rules.set('msg', 'ws "<|message:" recipient ">" "\\n" msg-body')

    const requiredRecipient = this.config.protocol.requiredMessageTo
    if (requiredRecipient !== null) {
      rules.set('forced-msg', `ws "<|message:${requiredRecipient}>" "\\n" msg-body`)
    }
  }

  private addYieldRules(rules: RuleMap): void {
    const alternatives = this.config.protocol.yieldTags
      .map(target => `ws "<|yield:${target}|>"`)
      .join(' | ')
    rules.set('yield', alternatives)
  }

  private addInvokeCloseRule(rules: RuleMap): void {
    // Lenient invoke close — accepts all 4 close tag modes
    rules.set('invoke-end', '"<invoke|>" | "</invoke|>" | "</invoke>" | "<invoke>"')
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

    if (recipient !== null) {
      const lensCount = maxLenses ?? 6
      const lensSlots: string[] = []
      for (let i = 0; i < lensCount; i++) {
        lensSlots.push('lens-cap?')
      }
      parts.push(...lensSlots)
    } else if (this.config.protocol.minLenses === 1) {
      parts.push('lens+')
    } else {
      parts.push('lens*')
    }

    if (recipient !== null) {
      parts.push('forced-msg')
    }

    parts.push(this.buildMiddleAlternative())
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
 * No `""` exits — the body MUST terminate by consuming a close tag.
 * The DFA is the sole handler of close tag detection; outer rules
 * do not include literal close tags.
 *
 * @param prefix - Rule name prefix (e.g., "think", "msg", "param", "filter")
 * @param tagName - The tag name for the close delimiter (e.g., "think", "message", "parameter", "filter")
 * @returns Array of GBNF rule strings
 */
export function generateBodyRules(prefix: string, tagName: string): string[] {
  const rules: string[] = []
  const n = tagName.length
  const s = (k: number) => `${prefix}-body-s${k}`
  const slashState = `${prefix}-body-sl`
  const pipeState = `${prefix}-body-pp`

  rules.push(`${prefix}-body ::= ${s(0)}`)

  // s0: base state — accumulate non-'<' content, or start matching on '<'
  // No "" exit — body must end with close tag
  rules.push(`${s(0)} ::= [^<] ${s(0)} | "<" ${s(1)}`)

  // s1: after '<' — accept '/' (slash variant) or first tagname char (canonical)
  const fc = escapeGbnfChar(tagName[0])
  const fcClass = escapeGbnfCharClass(tagName[0])
  rules.push(`${s(1)} ::= "/" ${slashState} | ${fc} ${s(2)} | "<" ${s(1)} | [^</${fcClass}] ${s(0)}`)

  // slashState: after '</' — expect first tagname char (shared path)
  rules.push(`${slashState} ::= ${fc} ${s(2)} | "<" ${s(1)} | [^<${fcClass}] ${s(0)}`)

  // s2..s{n}: match remaining tagname chars
  for (let i = 1; i < n; i++) {
    const ch = tagName[i]
    const esc = escapeGbnfChar(ch)
    const ccExclude = ch === '-' ? '[^<-]' : `[^${escapeGbnfCharClass(ch)}<]`
    rules.push(`${s(i + 1)} ::= ${esc} ${s(i + 2)} | "<" ${s(1)} | ${ccExclude} ${s(0)}`)
  }

  // Trailing whitespace states after close tag '>'
  const tw0 = `${prefix}-body-tw0`
  const tw1 = `${prefix}-body-tw1`
  const tw2 = `${prefix}-body-tw2`

  // s{n+1}: after full tagname — accept '|' (pipe variants) or '>' (no-pipe variants → trailing ws)
  rules.push(`${s(n + 1)} ::= "|" ${pipeState} | ">" ${tw0} | "<" ${s(1)} | [^<|>] ${s(0)}`)

  // pipeState: after '<[/]tagname|' — accept '>' to close (→ trailing ws)
  rules.push(`${pipeState} ::= ">" ${tw0} | "<" ${s(1)} | [^<>] ${s(0)}`)

  // tw0..tw2: trailing whitespace after close tag — 0-2 spaces/tabs then mandatory newline
  rules.push(`${tw0} ::= [ \\t] ${tw1} | "\\n"`)
  rules.push(`${tw1} ::= [ \\t] ${tw2} | "\\n"`)
  rules.push(`${tw2} ::= "\\n"`)

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

  // Generate parameter rules — each references shared param-body DFA
  const paramAlts: string[] = []
  for (const param of parameters) {
    const paramRuleName = `${ruleName}-p-${sanitizeParamName(param.name)}`
    paramAlts.push(paramRuleName)
    // DFA handles close tag — no literal "<parameter|>" in outer rule
    rules.push(`${paramRuleName} ::= "<|parameter:${param.name}>" param-body`)
  }

  // Tool invoke rule — unordered parameters, bounded to N occurrences (one per param)
  // "\n" after open tag consumed here (before param/close choice point)
  // Each param slot has hws for indentation
  const paramAlt = paramAlts.length > 0 ? `(bhws (${paramAlts.join(' | ')}))?` : ''
  const paramSeq = paramAlts.length > 0 ? Array(paramAlts.length).fill(paramAlt).join(' ') : ''
  
  rules.push(`${ruleName} ::= ws "<|${toolKeyword}:${tagName}>" "\\n" ${paramSeq} bhws ${ruleName}-close`)
  
  // Close alternatives — NO leading "\n" (consumed by open rule or previous param's DFA tw states)
  rules.push(`${ruleName}-close ::= invoke-end bhws "\\n" | "<invoke|filter>" "\\n" filter-body`)

  return rules
}

// =============================================================================
// Utilities
// =============================================================================

export function sanitizeRuleName(tagName: string): string {
  return `t-${tagName.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}`
}

export function sanitizeParamName(name: string): string {
  return name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()
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
