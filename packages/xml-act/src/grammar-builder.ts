
import { buildToolRules, generateBodyRules, type GrammarToolDef } from './grammar-shared'
import { LEAD_YIELD_TAGS } from './constants'

export { generateBodyRules, type GrammarToolDef } from './grammar-shared'

// =============================================================================
// Types
// =============================================================================

export interface ProtocolConfig {
  readonly minLenses: 0 | 1
  readonly allowMessages: boolean
  readonly allowTools: boolean
  readonly requiredMessageTo: string | null
  /** Maximum number of lenses allowed when a forced message is required.
   *  Prevents infinite lens loops by capping the model's thinking space. */
  readonly maxLenses: number | undefined
  /** Yield tags to use at turn end. Defaults to lead yield tags. */
  readonly yieldTags: ReadonlyArray<string>
}

export interface GrammarConfig {
  readonly tools: ReadonlyArray<GrammarToolDef>
  readonly protocol: ProtocolConfig
}

export interface GrammarBuildOptions {
  readonly minLenses?: 0 | 1
  readonly requiredMessageTo?: string
  /** Maximum number of lenses allowed when a forced message is required.
   *  Prevents infinite lens loops by capping the model's thinking space. */
  readonly maxLenses?: number
  /** Yield tags to use at turn end. Defaults to lead yield tags. */
  readonly yieldTags?: ReadonlyArray<string>
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

  withOptions(options: GrammarBuildOptions): GrammarBuilder {
    let next = this as GrammarBuilder
    if (options.minLenses !== undefined) next = next.withMinLenses(options.minLenses)
    if (options.requiredMessageTo !== undefined) next = next.requireMessageTo(options.requiredMessageTo)
    if (options.maxLenses !== undefined) next = next.withMaxLenses(options.maxLenses)
    if (options.yieldTags !== undefined) next = next.withYieldTags(options.yieldTags)
    return next
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
    // ws: unbounded whitespace between elements (normal case)
    // ws1: required whitespace (for separating attributes)
    // ws-bounded: limited whitespace (1-4 chars) used between lenses when a forced
    //   element follows, to prevent the model from looping on whitespace tokens
    rules.set('ws', '[ \\t\\n]*')
    rules.set('ws1', '[ \\t\\n]+')
    rules.set('ws-bounded', '[ \\t\\n] [ \\t\\n]? [ \\t\\n]? [ \\t\\n]?')
  }

  private addLensRules(rules: RuleMap): void {
    // lens: a thinking lens with unbounded trailing whitespace (used when no forced element follows)
    // lens-tight: same but with bounded whitespace, used in capped lens slots before a forced
    //   message to prevent infinite whitespace loops
    rules.set('lens', '"<lens name=\\"" lensname "\\">" lens-body "</lens>" ws')
    rules.set('lens-tight', '"<lens name=\\"" lensname "\\">" lens-body "</lens>" ws-bounded')
    rules.set('lensname', '"alignment" | "tasks" | "diligence" | "skills" | "turn" | "pivot"')

    for (const rule of generateBodyRules('lens', 'lens')) {
      const match = rule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }
  }

  private addMessageRules(rules: RuleMap): void {
    rules.set('msg', '"<message to=\\"" ([^"] | "\\\\\\"")*  "\\">" msg-body "</message>" ws')

    for (const rule of generateBodyRules('msg', 'message')) {
      const match = rule.match(/^(\S+) ::= (.+)$/)
      if (match) rules.set(match[1], match[2])
    }

    const recipient = this.config.protocol.requiredMessageTo
    if (recipient !== null) {
      rules.set('forced-msg', `"<message to=\\"${recipient}\\">" msg-body "</message>" ws`)
    }
  }

  private addYieldRules(rules: RuleMap): void {
    const alternatives = this.config.protocol.yieldTags
      .map(tag => `"<${tag}/>"`)
      .join(' | ')
    rules.set('yield', alternatives)
  }

  private addToolRules(rules: RuleMap): void {
    const { toolRule, rules: toolRuleStrings } = buildToolRules(this.config.tools)

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
    //
    // When a forced message is required (e.g., user reply), we cap the number
    // of lenses to prevent infinite loops. The model can use up to maxLenses
    // thinking lenses, but then MUST emit the forced message before continuing.

    // Lens section
    if (recipient !== null) {
      // Forced message case: cap lenses at maxLenses to prevent infinite loops.
      // Uses lens-tight (bounded whitespace) to prevent the model from looping
      // on whitespace tokens between lenses or before the forced message.
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
    if (this.config.protocol.allowTools) middle.push('tool')
    return middle.length > 0 ? `(${middle.join(' | ')})*` : 'msg*'
  }
}
