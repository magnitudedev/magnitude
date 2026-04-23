
import { LEAD_YIELD_TAGS } from '../constants'
import { VALID_CHILDREN } from '../nesting'

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
  /** Available lens names (kept for metadata; no longer affects grammar generation). */
  readonly lensNames: ReadonlyArray<string>
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

function addRule(rules: RuleMap, line: string): void {
  const match = line.match(/^(\S+) ::= (.+)$/)
  if (match) rules.set(match[1], match[2])
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

  withOptions(options: GrammarBuildOptions): GrammarBuilder {
    let next = this as GrammarBuilder
    if (options.minLenses !== undefined) next = next.withMinLenses(options.minLenses)
    if (options.requiredMessageTo !== undefined) next = next.requireMessageTo(options.requiredMessageTo)
    if (options.maxLenses !== undefined) next = next.withMaxLenses(options.maxLenses)
    if (options.yieldTags !== undefined) next = next.withYieldTags(options.yieldTags)
    if (options.lensNames !== undefined) next = next.withLensNames(options.lensNames)
    return next
  }

  build(): string {
    const { requiredMessageTo, maxLenses } = this.config.protocol

    // Validation
    if (maxLenses !== undefined && requiredMessageTo === null) {
      throw new Error('maxLenses requires requiredMessageTo to be set')
    }

    const rules: RuleMap = new Map()

    this.addWhitespaceRules(rules)
    this.addAttributeRules(rules)
    this.addYieldRules(rules)
    this.addContinuationRules(rules)
    this.addSharedBucRules(rules)
    this.addTopLevelBodyRules(rules)
    this.addToolRules(rules)
    this.addRootRule(rules)

    return serializeGrammar(rules)
  }

  // ---------------------------------------------------------------------------
  // Rule contributors
  // ---------------------------------------------------------------------------

  private addWhitespaceRules(rules: RuleMap): void {
    rules.set('ws', '[ \\t\\n]*')
  }

  private addAttributeRules(rules: RuleMap): void {
    rules.set('quoted-value', '[^"]*')
    rules.set('reason-attrs', '" about=\\"" quoted-value "\\""')
    rules.set('reason-attrs-opt', 'reason-attrs | ""')
    rules.set('msg-attrs', '" to=\\"" quoted-value "\\""')
  }

  private addYieldRules(rules: RuleMap): void {
    const yieldTags = this.config.protocol.yieldTags
    const withLt = yieldTags.map(t => `"<${t}/>"`)
    const noLt = yieldTags.map(t => `"${t}/>"`)
    rules.set('yield', withLt.join(' | '))
    rules.set('yield-no-lt', noLt.join(' | '))
  }

  private addContinuationRules(rules: RuleMap): void {
    const { allowMessages, allowTools } = this.config.protocol
    const proseChildren = VALID_CHILDREN.prose

    // Post-lens phase: message and/or invoke, then yield
    const postItems: string[] = []
    const postItemsNoLt: string[] = []
    for (const child of proseChildren) {
      if (child === 'magnitude:reason') continue
      if (child === 'magnitude:message' && allowMessages) {
        postItems.push('"<magnitude:message" msg-attrs ">" msg-body-s0')
        postItemsNoLt.push('"magnitude:message" msg-attrs ">" msg-body-s0')
      } else if (child === 'magnitude:invoke' && allowTools) {
        postItems.push('"<magnitude:invoke" invoke-attrs ">" invoke-body')
        postItemsNoLt.push('"magnitude:invoke" invoke-attrs ">" invoke-body')
      }
    }

    const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"<magnitude:message" msg-attrs ">" msg-body-s0'
    const postNoLtItems = postItemsNoLt.length > 0 ? postItemsNoLt : ['"magnitude:message" msg-attrs ">" msg-body-s0']

    rules.set('turn-item-post', postItemRule)
    rules.set('turn-next-post', 'ws turn-item-post | ws yield')
    rules.set('turn-next-post-no-lt', [...postNoLtItems, 'yield-no-lt'].join(' | '))

    // Lens phase: reason + post-lens items
    const hasReason = (proseChildren as readonly string[]).includes('magnitude:reason')
    const lensItems = hasReason
      ? ['"<magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItems]
      : postItems
    const lensItemsNoLt = hasReason
      ? ['"magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItemsNoLt]
      : postItemsNoLt

    rules.set('turn-item-lens', lensItems.join(' | '))
    rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
    rules.set('turn-next-lens-no-lt', [...lensItemsNoLt, 'yield-no-lt'].join(' | '))
  }

  /**
   * Shared BUC (body-until-close) rules for each close tag name.
   * These are reused across all body rules for the same tag.
   */
  private addSharedBucRules(rules: RuleMap): void {
    // param-buc: excludes </magnitude:parameter>
    for (const rule of generateBucRules('param-buc', 'magnitude:parameter')) {
      addRule(rules, rule)
    }
    // filter-buc: excludes </magnitude:filter>
    for (const rule of generateBucRules('filter-buc', 'magnitude:filter')) {
      addRule(rules, rule)
    }
    // reason-buc: excludes </magnitude:reason>
    for (const rule of generateBucRules('reason-buc', 'magnitude:reason')) {
      addRule(rules, rule)
    }
    // msg-buc: excludes </magnitude:message>
    for (const rule of generateBucRules('msg-buc', 'magnitude:message')) {
      addRule(rules, rule)
    }
  }

  /**
   * Top-level body rules using recursive greedy last-match.
   * Confirmation: </tagname> + ws + < (next structural tag).
   */
  private addTopLevelBodyRules(rules: RuleMap): void {
    // reason body: greedy last-match, confirmed by ws + next lens-phase tag
    const reasonClose = '"</magnitude:reason>"'
    rules.set('reason-body-s0',
      `reason-buc (${reasonClose} reason-buc)* ${reasonClose} ws turn-item-lens-no-lt-or-yield`)

    // msg body: greedy last-match, confirmed by ws + next post-phase tag
    const msgClose = '"</magnitude:message>"'
    rules.set('msg-body-s0',
      `msg-buc (${msgClose} msg-buc)* ${msgClose} ws turn-item-post-no-lt-or-yield`)

    // Helper rules: the continuation after close + ws must start with <
    // which is consumed by the no-lt variants, OR be a yield (which starts with <)
    rules.set('turn-item-lens-no-lt-or-yield', 'turn-item-lens | yield')
    rules.set('turn-item-post-no-lt-or-yield', 'turn-item-post | yield')
  }

  /**
   * Per-tool grammar rules with constrained param names, bounded counts,
   * and position-aware greedy matching.
   */
  private addToolRules(rules: RuleMap): void {
    const tools = this.config.tools

    if (tools.length === 0) {
      // Fallback: generic invoke with free-form tool name and params
      rules.set('invoke-attrs', '" tool=\\"" quoted-value "\\""')
      rules.set('invoke-body', 'ws invoke-generic-item | ws "</magnitude:invoke>" turn-next-post')
      rules.set('invoke-generic-item',
        '"<magnitude:parameter" " name=\\"" quoted-value "\\"" ">" generic-param-body-s0 | "<magnitude:filter>" generic-filter-body-s0')
      // Generic param body: greedy last-match, confirmed by next invoke child or close
      rules.set('generic-param-body-s0',
        'param-buc ("</magnitude:parameter>" param-buc)* "</magnitude:parameter>" (ws invoke-generic-item | ws "</magnitude:invoke>" turn-next-post)')
      rules.set('generic-filter-body-s0',
        'filter-buc ("</magnitude:filter>" filter-buc)* "</magnitude:filter>" ws "</magnitude:invoke>" turn-next-post')
      return
    }

    // Build invoke-attrs as enumerated tool names
    const toolNameAlts = tools.map(t => `" tool=\\"${escapeGbnfString(t.tagName)}\\""`)
    rules.set('invoke-attrs', toolNameAlts.join(' | '))

    // Build invoke-body as dispatch to per-tool rules
    // After <invoke tool="X">, we need to dispatch based on tool name.
    // Since the tool name is already consumed as an attribute, we use per-tool invoke rules.
    // We restructure: instead of generic invoke-body, each tool gets its own invoke rule.

    // Rewrite: the continuation rules reference invoke-body which is called after
    // <invoke invoke-attrs ">". We need invoke-body to dispatch per tool.
    // But GBNF doesn't have conditional dispatch on previously consumed content.
    //
    // Solution: instead of one invoke-attrs + invoke-body, generate per-tool alternatives
    // in the continuation rules directly.

    // Build per-tool invoke alternatives
    const invokeAlts: string[] = []
    const invokeAltsNoLt: string[] = []

    for (const tool of tools) {
      const safeName = sanitizeRuleName(tool.tagName)
      const toolAttr = `" tool=\\"${escapeGbnfString(tool.tagName)}\\"">`

      this.addPerToolRules(rules, tool, safeName)

      invokeAlts.push(`"<magnitude:invoke" " tool=\\"${escapeGbnfString(tool.tagName)}\\"" ">" ${safeName}-body`)
      invokeAltsNoLt.push(`"magnitude:invoke" " tool=\\"${escapeGbnfString(tool.tagName)}\\"" ">" ${safeName}-body`)
    }

    // Override the invoke entries in continuation rules
    // We need to replace the generic invoke references with per-tool alternatives
    // Rebuild turn-item-post and turn-item-lens with per-tool invoke alts

    const { allowMessages } = this.config.protocol
    const proseChildren = VALID_CHILDREN.prose

    const postItems: string[] = []
    const postItemsNoLt: string[] = []
    for (const child of proseChildren) {
      if (child === 'magnitude:reason') continue
      if (child === 'magnitude:message' && allowMessages) {
        postItems.push('"<magnitude:message" msg-attrs ">" msg-body-s0')
        postItemsNoLt.push('"magnitude:message" msg-attrs ">" msg-body-s0')
      } else if (child === 'magnitude:invoke') {
        postItems.push(...invokeAlts)
        postItemsNoLt.push(...invokeAltsNoLt)
      }
    }

    const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"<magnitude:message" msg-attrs ">" msg-body-s0'
    const postNoLtItems = postItemsNoLt.length > 0 ? postItemsNoLt : ['"magnitude:message" msg-attrs ">" msg-body-s0']

    // Override the rules set by addContinuationRules
    rules.set('turn-item-post', postItemRule)
    rules.set('turn-next-post', 'ws turn-item-post | ws yield')
    rules.set('turn-next-post-no-lt', [...postNoLtItems, 'yield-no-lt'].join(' | '))

    const hasReason = (proseChildren as readonly string[]).includes('magnitude:reason')
    const lensItems = hasReason
      ? ['"<magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItems]
      : postItems
    const lensItemsNoLt = hasReason
      ? ['"magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItemsNoLt]
      : postItemsNoLt

    rules.set('turn-item-lens', lensItems.join(' | '))
    rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
    rules.set('turn-next-lens-no-lt', [...lensItemsNoLt, 'yield-no-lt'].join(' | '))

    // Re-derive the helper rules for top-level body confirmation
    rules.set('turn-item-lens-no-lt-or-yield', 'turn-item-lens | yield')
    rules.set('turn-item-post-no-lt-or-yield', 'turn-item-post | yield')
  }

  /**
   * Generate per-tool rules: param name constraints, bounded count,
   * position-aware greedy body rules.
   */
  private addPerToolRules(rules: RuleMap, tool: GrammarToolDef, safeName: string): void {
    const N = tool.parameters.length

    if (N === 0) {
      // 0-param tool: invoke body is just ws + close
      rules.set(`${safeName}-body`, `ws "</magnitude:invoke>" turn-next-post`)
      return
    }

    // Constrained param names for this tool
    const paramNameAlts = tool.parameters.map(p =>
      `" name=\\"${escapeGbnfString(p.name)}\\""`)
    rules.set(`${safeName}-param-names`, paramNameAlts.join(' | '))

    // Generate sequence chain: seq-N down to seq-1
    // seq-K means K parameter slots remaining
    for (let k = N; k >= 1; k--) {
      const seqName = `${safeName}-seq-${k}`
      const isLastSlot = k === 1

      // Parameter open with constrained names, chaining to position-specific body
      const bodyRule = isLastSlot ? `${safeName}-last-body-s0` : `${safeName}-nonlast-body-s0-${k}`
      const paramAlt = `ws "<magnitude:parameter" ${safeName}-param-names ">" ${bodyRule}`
      const filterAlt = `ws "<magnitude:filter>" ${safeName}-filter-body-s0`
      const closeAlt = `ws "</magnitude:invoke>" turn-next-post`

      rules.set(seqName, [paramAlt, filterAlt, closeAlt].join(' | '))
    }

    // Non-last body rules: for each position K > 1, body chains to seq-(K-1)
    for (let k = N; k >= 2; k--) {
      const nextSeq = `${safeName}-seq-${k - 1}`
      rules.set(`${safeName}-nonlast-body-s0-${k}`,
        `param-buc ("</magnitude:parameter>" param-buc)* "</magnitude:parameter>" ${nextSeq}`)
    }

    // Last body rule: deep confirmation through invoke close + next top-level tag
    rules.set(`${safeName}-last-body-s0`,
      `param-buc ("</magnitude:parameter>" param-buc)* "</magnitude:parameter>" ws "</magnitude:invoke>" turn-next-post`)

    // Filter body: always deep (filter closes invoke)
    rules.set(`${safeName}-filter-body-s0`,
      `filter-buc ("</magnitude:filter>" filter-buc)* "</magnitude:filter>" ws "</magnitude:invoke>" turn-next-post`)

    // Entry point: invoke body starts at seq-N
    rules.set(`${safeName}-body`, `${safeName}-seq-${N}`)
  }

  private addRootRule(rules: RuleMap): void {
    const { minLenses, requiredMessageTo, maxLenses } = this.config.protocol

    if (requiredMessageTo !== null) {
      this.addForcedMessageRules(rules, requiredMessageTo, maxLenses)
    } else if (minLenses === 1) {
      rules.set('root', 'ws "<magnitude:reason" reason-attrs-opt ">" reason-body-s0')
    } else {
      rules.set('root', 'turn-next-lens')
    }
  }

  private addForcedMessageRules(rules: RuleMap, recipient: string, maxLenses: number | undefined): void {
    const escapedRecipient = recipient.replace(/"/g, '\\"')
    rules.set('forced-msg', `"<magnitude:message to=\\"${escapedRecipient}\\">" msg-body-s0`)
    rules.set('forced-msg-no-lt', `"magnitude:message to=\\"${escapedRecipient}\\">" msg-body-s0`)

    if (maxLenses !== undefined) {
      for (let k = maxLenses; k >= 0; k--) {
        if (k === 0) {
          rules.set(`turn-next-forced-0`, 'ws forced-msg')
          rules.set(`turn-next-forced-0-no-lt`, 'forced-msg-no-lt')
        } else {
          const nextK = k - 1
          // Reason body for forced phase — greedy last-match, chains to next forced level
          const reasonClose = '"</magnitude:reason>"'
          rules.set(`reason-forced-${k}-body-s0`,
            `reason-buc (${reasonClose} reason-buc)* ${reasonClose} turn-next-forced-${nextK}`)
          rules.set(
            `turn-next-forced-${k}`,
            `ws "<magnitude:reason" reason-attrs-opt ">" reason-forced-${k}-body-s0 | ws forced-msg`
          )
          rules.set(
            `turn-next-forced-${k}-no-lt`,
            `"magnitude:reason" reason-attrs-opt ">" reason-forced-${k}-body-s0 | forced-msg-no-lt`
          )
        }
      }
      rules.set('root', `turn-next-forced-${maxLenses}`)
    } else {
      const reasonClose = '"</magnitude:reason>"'
      rules.set('reason-forced-body-s0',
        `reason-buc (${reasonClose} reason-buc)* ${reasonClose} turn-next-forced`)
      rules.set(
        'turn-next-forced',
        'ws "<magnitude:reason" reason-attrs-opt ">" reason-forced-body-s0 | ws forced-msg'
      )
      rules.set(
        'turn-next-forced-no-lt',
        '"magnitude:reason" reason-attrs-opt ">" reason-forced-body-s0 | forced-msg-no-lt'
      )
      rules.set('root', 'turn-next-forced')
    }
  }
}

// =============================================================================
// BUC (Body-Until-Close) Rule Generation
// =============================================================================

/**
 * Generate BUC exclusion rules for a given close tag.
 * BUC matches any string NOT containing the close tag `</tagName>`.
 *
 * The pattern: for each prefix of the close tag, match that prefix
 * followed by a character that breaks the pattern.
 */
export function generateBucRules(prefix: string, tagName: string): string[] {
  const lines: string[] = []
  const closeTag = '</' + tagName + '>'
  const closeChars = closeTag.split('')

  const alts: string[] = []

  // First alt: any char that doesn't start the close tag
  alts.push(`[^${escapeGbnfCharClass(closeChars[0])}]`)

  // Subsequent alts: match prefix of close tag, then a char that breaks it
  let pfx = ''
  for (let i = 0; i < closeChars.length - 1; i++) {
    pfx += closeChars[i]
    const nextChar = closeChars[i + 1]
    const nextCharEsc = escapeGbnfCharClass(nextChar)
    alts.push(`"${pfx}" [^${nextCharEsc}]`)
  }

  lines.push(`${prefix} ::= (${alts.join(' | ')})*`)
  return lines
}

// =============================================================================
// Utilities
// =============================================================================

export function escapeGbnfChar(ch: string): string {
  switch (ch) {
    case '"': return '\\"'
    case '\\': return '"\\\\"'
    case '\n': return '"\\n"'
    case '\t': return '"\\t"'
    case '<': return '"<"'
    case '>': return '">"'
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

export function escapeGbnfString(s: string): string {
  return s.replace(/"/g, '\\"').replace(/\\/g, '\\\\')
}

export function sanitizeRuleName(tagName: string): string {
  return `t-${tagName.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}`
}

export function sanitizeParamName(name: string): string {
  return name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()
}

// =============================================================================
// Legacy exports (kept for API compatibility)
// =============================================================================

/**
 * @deprecated Use GrammarBuilder directly. This is kept for existing callers.
 */
export interface BodyContext {
  readonly confirmRule: string
  readonly confirmNoLtRule: string
}

/**
 * @deprecated Replaced by recursive greedy body rules.
 */
export function generateBodyRules(prefix: string, tagName: string, context: BodyContext): string[] {
  // Legacy: generate DFA body rules with tw states
  // This is no longer used by the builder but may be referenced by tests
  const lines: string[] = []
  const L = tagName.length
  const MAX_TW = 4

  lines.push(`${prefix} ::= ${prefix}-s0`)
  lines.push(`${prefix}-s0 ::= [^<] ${prefix}-s0 | "<" ${prefix}-s1`)
  lines.push(`${prefix}-s1 ::= "/" ${prefix}-sl | "<" ${prefix}-s1 | [^/<] ${prefix}-s0`)

  const fc = tagName[0]
  const fcEsc = escapeGbnfCharClass(fc)

  if (L === 1) {
    const gtState = `${prefix}-gt`
    lines.push(`${prefix}-sl ::= "${fc}" ${gtState} | "<" ${prefix}-s1 | [^<${fcEsc}] ${prefix}-s0`)
    lines.push(`${gtState} ::= ">" ${prefix}-tw0 | "<" ${prefix}-s1 | [^<>] ${prefix}-s0`)
  } else {
    lines.push(`${prefix}-sl ::= "${fc}" ${prefix}-s2 | "<" ${prefix}-s1 | [^<${fcEsc}] ${prefix}-s0`)
    for (let i = 2; i <= L; i++) {
      const ch = tagName[i - 1]
      const chEsc = escapeGbnfCharClass(ch)
      const nextState = i === L ? `${prefix}-gt` : `${prefix}-s${i + 1}`
      lines.push(`${prefix}-s${i} ::= "${ch}" ${nextState} | "<" ${prefix}-s1 | [^<${chEsc}] ${prefix}-s0`)
    }
    lines.push(`${prefix}-gt ::= ">" ${prefix}-tw0 | "<" ${prefix}-s1 | [^<>] ${prefix}-s0`)
  }

  for (let i = 0; i <= MAX_TW; i++) {
    const stateName = `${prefix}-tw${i}`
    if (i < MAX_TW) {
      lines.push(
        `${stateName} ::= [ \\t] ${prefix}-tw${i + 1}` +
        ` | "\\n" ${context.confirmRule}` +
        ` | "<" ${context.confirmNoLtRule}` +
        ` | [^ \\t\\n<] ${prefix}-s0`
      )
    } else {
      lines.push(
        `${stateName} ::= "\\n" ${context.confirmRule}` +
        ` | "<" ${context.confirmNoLtRule}` +
        ` | [^ \\n<] ${prefix}-s0`
      )
    }
  }

  return lines
}

/**
 * @deprecated Replaced by per-tool body rules with position-aware continuations.
 */
export function generateRecursiveBodyRules(prefix: string, tagName: string, continuationRule: string): string[] {
  const lines: string[] = []
  const closeTag = '</' + tagName + '>'
  const closeChars = closeTag.split('')

  const alts: string[] = []
  alts.push(`[^${escapeGbnfCharClass(closeChars[0])}]`)

  let pfx = ''
  for (let i = 0; i < closeChars.length - 1; i++) {
    pfx += closeChars[i]
    const nextChar = closeChars[i + 1]
    const nextCharEsc = escapeGbnfCharClass(nextChar)
    alts.push(`"${pfx}" [^${nextCharEsc}]`)
  }

  lines.push(`${prefix}-buc ::= (${alts.join(' | ')})*`)
  const closeLiteral = `"${closeTag}"`
  lines.push(`${prefix}-s0 ::= ${prefix}-buc (${closeLiteral} ${prefix}-buc)* ${closeLiteral} ${continuationRule}`)

  return lines
}
