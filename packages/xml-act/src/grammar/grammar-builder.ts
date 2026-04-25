
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
    rules.set('msg-attrs-opt', 'msg-attrs | ""')
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
        postItems.push('"<magnitude:message" msg-attrs-opt ">" msg-body-s0')
        postItemsNoLt.push('"magnitude:message" msg-attrs-opt ">" msg-body-s0')
      } else if (child === 'magnitude:invoke' && allowTools) {
        postItems.push('"<magnitude:invoke" invoke-attrs ">" invoke-body')
        postItemsNoLt.push('"magnitude:invoke" invoke-attrs ">" invoke-body')
      }
    }

    const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"<magnitude:message" msg-attrs-opt ">" msg-body-s0'
    const postNoLtItems = postItemsNoLt.length > 0 ? postItemsNoLt : ['"magnitude:message" msg-attrs-opt ">" msg-body-s0']

    postItems.push('"<magnitude:escape>" escape-body-post-s0')
    postItemsNoLt.push('"magnitude:escape>" escape-body-post-s0')

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

    lensItems.push('"<magnitude:escape>" escape-body-lens-s0')
    lensItemsNoLt.push('"magnitude:escape>" escape-body-lens-s0')

    rules.set('turn-item-lens', lensItems.join(' | '))
    rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
    rules.set('turn-next-lens-no-lt', [...lensItemsNoLt, 'yield-no-lt'].join(' | '))
  }

  /**
   * Shared BUC (body-until-close) rules for each close tag name.
   * These are reused across all body rules for the same tag.
   */
  private addSharedBucRules(rules: RuleMap): void {
    const escapeTag = 'magnitude:escape'
    const magnitudePrefix = 'magnitude:'

    // Plain BUC rules (exclude only their close tag)
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
    // escape-buc: excludes </magnitude:escape> (no compound needed — escape doesn't nest)
    for (const rule of generateBucRules('escape-buc', escapeTag)) {
      addRule(rules, rule)
    }

    // Compound BUC rules for param/filter: exclude close tag AND <magnitude: open prefix
    for (const rule of generateCompoundBucRules('param-esc-buc', 'magnitude:parameter', magnitudePrefix, { excludeOpenPrefix: true })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('filter-esc-buc', 'magnitude:filter', magnitudePrefix, { excludeOpenPrefix: true })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('reason-esc-buc', 'magnitude:reason', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:' })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('msg-esc-buc', 'magnitude:message', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:' })) {
      addRule(rules, rule)
    }

    // Line-aware BUC variants: also exclude \n so we can track line boundaries
    // Used in body rules that need newline-confirmed mismatch recovery
    for (const rule of generateCompoundBucRules('reason-line-buc', 'magnitude:reason', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', excludeChars: ['\n'] })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('msg-line-buc', 'magnitude:message', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', excludeChars: ['\n'] })) {
      addRule(rules, rule)
    }

    // Escape-aware body content rules: BUC interleaved with escape blocks
    // Pattern: buc (escape-block buc)* — no nested repetition ambiguity
    const escBlock = `"<${escapeTag}>" escape-buc "</${escapeTag}>"`
    // Magnitude open absorber: matches rest of any <magnitude:X...> or <magnitude:X.../> open tag
    // Used to consume false <magnitude:...> opens as content in greedy body patterns
    rules.set('mag-open-rest', '[^>]* ">"')

    // Magnitude close absorber: matches any </magnitude:X> close tag
    rules.set('mag-close-rest', '[a-z_:]* ">"')

    // Line-aware BUC variants for param/filter: also exclude \n and </magnitude: close prefix
    // Used in strict body rules to detect same-line mismatched closes
    for (const rule of generateCompoundBucRules('param-line-buc', 'magnitude:parameter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', excludeChars: ['\n'] })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('filter-line-buc', 'magnitude:filter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', excludeChars: ['\n'] })) {
      addRule(rules, rule)
    }

    // After-lt BUC variants: used after body-level '<' recovery.
    // These also stop at '/magnitude:parameter>' (close tag without leading '<').
    for (const rule of generateCompoundBucRules('param-alt-buc', 'magnitude:parameter', magnitudePrefix, { excludeOpenPrefix: true, afterLt: true })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('filter-alt-buc', 'magnitude:filter', magnitudePrefix, { excludeOpenPrefix: true, afterLt: true })) {
      addRule(rules, rule)
    }

    // Line-aware after-lt BUC variants for param/filter strict body
    for (const rule of generateCompoundBucRules('param-line-alt-buc', 'magnitude:parameter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', excludeChars: ['\n'], afterLt: true })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('filter-line-alt-buc', 'magnitude:filter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', excludeChars: ['\n'], afterLt: true })) {
      addRule(rules, rule)
    }

    // Strict body rules: BUC stops at <magnitude: prefix and at '<<'.
    // Body has '<' recovery that switches to after-lt BUC (which also stops at '/close>').
    // If an invalid <magnitude:...> open appears before any false close, the grammar rejects.
    const ltRecoveryParam = `"<" param-alt-buc`
    const ltRecoveryFilter = `"<" filter-alt-buc`

    // Shared newline + close absorber patterns
    const nlMagClose = `"\\n" "</magnitude:" mag-close-rest`
    const nlOnly = `"\\n"`

    // Strict body rules for param/filter: use line-aware BUCs to detect same-line mismatched closes
    // Pattern: line-buc (\n mag-close line-buc | \n line-buc | escape line-buc | < line-alt-buc ...)*
    const paramLineBuc = 'param-line-buc'
    const filterLineBuc = 'filter-line-buc'
    const ltRecoveryParamLine = `"<" param-line-alt-buc`
    const ltRecoveryFilterLine = `"<" filter-line-alt-buc`
    rules.set('param-esc-body', `${paramLineBuc} (${escBlock} ${paramLineBuc} | ${nlMagClose} ${paramLineBuc} | ${nlOnly} ${paramLineBuc} | ${ltRecoveryParamLine} (${escBlock} ${paramLineBuc} | ${nlMagClose} ${paramLineBuc} | ${nlOnly} ${paramLineBuc})*)*`)
    rules.set('filter-esc-body', `${filterLineBuc} (${escBlock} ${filterLineBuc} | ${nlMagClose} ${filterLineBuc} | ${nlOnly} ${filterLineBuc} | ${ltRecoveryFilterLine} (${escBlock} ${filterLineBuc} | ${nlMagClose} ${filterLineBuc} | ${nlOnly} ${filterLineBuc})*)*`)

    // Permissive body rules: used AFTER false closes in greedy patterns.
    // These absorb <magnitude:...> opens as content since after a false close, everything is content.
    const magOpenAbsorb = `"<magnitude:" mag-open-rest`
    rules.set('param-greedy-body', `param-esc-buc (${escBlock} param-esc-buc | ${magOpenAbsorb} param-esc-buc | ${ltRecoveryParam} (${escBlock} param-esc-buc | ${magOpenAbsorb} param-esc-buc)*)*`)
    rules.set('filter-greedy-body', `filter-esc-buc (${escBlock} filter-esc-buc | ${magOpenAbsorb} filter-esc-buc | ${ltRecoveryFilter} (${escBlock} filter-esc-buc | ${magOpenAbsorb} filter-esc-buc)*)*`)

    // Line-aware body rules for reason/message: support newline-confirmed mismatch recovery
    const reasonLineBuc = 'reason-line-buc'
    const msgLineBuc = 'msg-line-buc'
    rules.set('reason-esc-body', `${reasonLineBuc} (${escBlock} ${reasonLineBuc} | ${nlMagClose} ${reasonLineBuc} | ${nlOnly} ${reasonLineBuc})*`)
    rules.set('msg-esc-body', `${msgLineBuc} (${escBlock} ${msgLineBuc} | ${nlMagClose} ${msgLineBuc} | ${nlOnly} ${msgLineBuc})*`)
  }

  /**
   * Top-level body rules using recursive greedy last-match.
   * Confirmation: </tagname> + ws + < (next structural tag).
   */
  private addTopLevelBodyRules(rules: RuleMap): void {
    // reason body: greedy last-match with inline escape support
    // Uses non-ext body: <magnitude:...> opens are rejected (only escape is allowed)
    const reasonClose = '"</magnitude:reason>"'
    rules.set('reason-body-s0',
      `reason-esc-body (${reasonClose} reason-esc-body)* ${reasonClose} ws turn-item-lens-no-lt-or-yield`)

    // msg body: greedy last-match with inline escape support
    // Uses non-ext body: <magnitude:...> opens are rejected (only escape is allowed)
    const msgClose = '"</magnitude:message>"'
    rules.set('msg-body-s0',
      `msg-esc-body (${msgClose} msg-esc-body)* ${msgClose} ws turn-item-post-no-lt-or-yield`)

    // Helper rules: the continuation after close + ws must start with <
    // which is consumed by the no-lt variants, OR be a yield (which starts with <)
    rules.set('turn-item-lens-no-lt-or-yield', 'turn-item-lens | yield')
    rules.set('turn-item-post-no-lt-or-yield', 'turn-item-post | yield')

    // escape body (lens phase): first close ends block (no nesting, no greedy)
    rules.set('escape-body-lens-s0',
      `escape-buc "</magnitude:escape>" ws turn-item-lens-no-lt-or-yield`)

    // escape body (post phase): first close ends block (no nesting, no greedy)
    rules.set('escape-body-post-s0',
      `escape-buc "</magnitude:escape>" ws turn-item-post-no-lt-or-yield`)
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
        `param-esc-body (("</magnitude:parameter>" | "/magnitude:parameter>") param-greedy-body)* ("</magnitude:parameter>" | "/magnitude:parameter>") (ws invoke-generic-item | ws "</magnitude:invoke>" turn-next-post)`)
      rules.set('generic-filter-body-s0',
        `filter-esc-body (("</magnitude:filter>" | "/magnitude:filter>") filter-greedy-body)* ("</magnitude:filter>" | "/magnitude:filter>") (ws invoke-generic-item | ws "</magnitude:invoke>" turn-next-post)`)
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

      this.addPerToolRules(rules, tool, safeName)

      invokeAlts.push(`"<magnitude:invoke" " tool=\\"${escapeGbnfString(tool.tagName)}\\"" ">" ${safeName}-body`)
      invokeAltsNoLt.push(`"magnitude:invoke" " tool=\\"${escapeGbnfString(tool.tagName)}\\"" ">" ${safeName}-body`)
      invokeAlts.push(`"<magnitude:${escapeGbnfString(tool.tagName)}" ">" ${safeName}-alias-body`)
      invokeAltsNoLt.push(`"magnitude:${escapeGbnfString(tool.tagName)}" ">" ${safeName}-alias-body`)
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
        postItems.push('"<magnitude:message" msg-attrs-opt ">" msg-body-s0')
        postItemsNoLt.push('"magnitude:message" msg-attrs-opt ">" msg-body-s0')
      } else if (child === 'magnitude:invoke') {
        postItems.push(...invokeAlts)
        postItemsNoLt.push(...invokeAltsNoLt)
      }
    }

    const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"<magnitude:message" msg-attrs-opt ">" msg-body-s0'
    const postNoLtItems = postItemsNoLt.length > 0 ? postItemsNoLt : ['"magnitude:message" msg-attrs-opt ">" msg-body-s0']

    // Add escape alternatives to post items
    postItems.push('"<magnitude:escape>" escape-body-post-s0')
    postItemsNoLt.push('"magnitude:escape>" escape-body-post-s0')

    const postItemRuleWithEscape = postItems.join(' | ')
    const postNoLtWithEscape = [...postItemsNoLt]

    // Override the rules set by addContinuationRules
    rules.set('turn-item-post', postItemRuleWithEscape)
    rules.set('turn-next-post', 'ws turn-item-post | ws yield')
    rules.set('turn-next-post-no-lt', [...postNoLtWithEscape, 'yield-no-lt'].join(' | '))

    const hasReason = (proseChildren as readonly string[]).includes('magnitude:reason')
    // Build lens items from post items WITHOUT escape, then add escape with lens body
    const lensItems = hasReason
      ? ['"<magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItems.filter(i => !i.includes('escape'))]
      : postItems.filter(i => !i.includes('escape'))
    const lensItemsNoLt = hasReason
      ? ['"magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItemsNoLt.filter(i => !i.includes('escape'))]
      : postItemsNoLt.filter(i => !i.includes('escape'))

    // Add escape with lens-phase body rule
    lensItems.push('"<magnitude:escape>" escape-body-lens-s0')
    lensItemsNoLt.push('"magnitude:escape>" escape-body-lens-s0')

    rules.set('turn-item-lens', lensItems.join(' | '))
    rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
    rules.set('turn-next-lens-no-lt', [...lensItemsNoLt, 'yield-no-lt'].join(' | '))

    // Re-derive the helper rules for top-level body confirmation
    rules.set('turn-item-lens-no-lt-or-yield', 'turn-item-lens | yield')
    rules.set('turn-item-post-no-lt-or-yield', 'turn-item-post | yield')

    // escape body (lens phase): first close ends block (no nesting, no greedy)
    rules.set('escape-body-lens-s0',
      `escape-buc "</magnitude:escape>" ws turn-item-lens-no-lt-or-yield`)

    // escape body (post phase): first close ends block (no nesting, no greedy)
    rules.set('escape-body-post-s0',
      `escape-buc "</magnitude:escape>" ws turn-item-post-no-lt-or-yield`)
  }

  /**
   * Generate per-tool rules: param name constraints, bounded count,
   * position-aware greedy body rules.
   */
  private addPerToolRules(rules: RuleMap, tool: GrammarToolDef, safeName: string): void {
    const N = tool.parameters.length
    const invokeClose = '"</magnitude:invoke>"'
    const aliasInvokeClose = `"</magnitude:${escapeGbnfString(tool.tagName)}>"`
    const aliasInvokeCloseAlt = `${invokeClose} | ${aliasInvokeClose}`
    const escBlock = `"<magnitude:escape>" escape-buc "</magnitude:escape>"`

    const canonicalParamBodyRule = (k: number): string =>
      k === 1 ? `${safeName}-last-body-s0` : `${safeName}-nonlast-body-s0-${k}`

    const aliasParamBodyRule = (k: number): string =>
      k === 1 ? `${safeName}-alias-last-body-s0` : `${safeName}-alias-nonlast-body-s0-${k}`

    const aliasParamSpecificBodyRule = (k: number, pSafe: string, aliasInvoke: boolean): string => {
      if (k === 1) return aliasInvoke ? `${safeName}-alias-last-body-s0-${pSafe}` : `${safeName}-last-body-s0-${pSafe}`
      return aliasInvoke ? `${safeName}-alias-nonlast-body-s0-${k}-${pSafe}` : `${safeName}-nonlast-body-s0-${k}-${pSafe}`
    }

    const canonicalParamAlt = (bodyRule: string): string =>
      `ws "<magnitude:parameter" ${safeName}-param-names ">" ${bodyRule}`

    const aliasParamAlts = (k: number, aliasInvoke: boolean): string[] =>
      tool.parameters.map(param => {
        const pSafe = sanitizeParamName(param.name)
        return `ws "<magnitude:${escapeGbnfString(param.name)}>" ${aliasParamSpecificBodyRule(k, pSafe, aliasInvoke)}`
      })

    if (N === 0) {
      // 0-param tool: invoke body is just ws + close
      rules.set(`${safeName}-body`, `ws ${invokeClose} turn-next-post`)
      rules.set(`${safeName}-alias-body`, `ws (${aliasInvokeCloseAlt}) turn-next-post`)
      return
    }

    // Constrained param names for this tool
    const paramNameAlts = tool.parameters.map(p =>
      `" name=\\"${escapeGbnfString(p.name)}\\""`)
    rules.set(`${safeName}-param-names`, paramNameAlts.join(' | '))

    for (const param of tool.parameters) {
      const pSafe = sanitizeParamName(param.name)

      for (const rule of generateCompoundBucRules(
        `${safeName}-${pSafe}-esc-buc`,
        `magnitude:${param.name}`,
        'magnitude:',
        { excludeOpenPrefix: true },
      )) {
        addRule(rules, rule)
      }

      const aliasBuc = `${safeName}-${pSafe}-esc-buc`
      // Generate after-lt variant for this alias BUC
      for (const rule of generateCompoundBucRules(
        `${safeName}-${pSafe}-alt-buc`,
        `magnitude:${param.name}`,
        'magnitude:',
        { excludeOpenPrefix: true, afterLt: true },
      )) {
        addRule(rules, rule)
      }
      const aliasAltBuc = `${safeName}-${pSafe}-alt-buc`
      const aliasLtRecovery = `"<" ${aliasAltBuc}`
      rules.set(
        `${safeName}-${pSafe}-esc-body`,
        `${aliasBuc} (${escBlock} ${aliasBuc} | ${aliasLtRecovery} (${escBlock} ${aliasBuc})*)*`,
      )
      // Greedy variant: also absorbs <magnitude:...> opens as content after false closes
      const magOpen = `"<magnitude:" mag-open-rest`
      rules.set(
        `${safeName}-${pSafe}-greedy-body`,
        `${aliasBuc} (${escBlock} ${aliasBuc} | ${magOpen} ${aliasBuc} | ${aliasLtRecovery} (${escBlock} ${aliasBuc} | ${magOpen} ${aliasBuc})*)*`,
      )
    }

    // Magnitude open absorber: consumes false <magnitude:...> opens as content after false closes in greedy patterns
    // Generate sequence chain: seq-N down to seq-1
    // seq-K means K parameter slots remaining
    for (let k = N; k >= 1; k--) {
      const canonicalBodyRule = canonicalParamBodyRule(k)
      const aliasBodyRule = aliasParamBodyRule(k)
      const canonicalParam = canonicalParamAlt(canonicalBodyRule)
      const aliasCanonicalParams = aliasParamAlts(k, false)
      const aliasInvokeParams = aliasParamAlts(k, true)
      // Filter chains back to same seq-K (filter doesn't consume a parameter slot)
      const canonicalFilterAlt = `ws "<magnitude:filter>" ${safeName}-filter-cont-body-s0-${k}`
      const aliasFilterAlt = `ws "<magnitude:filter>" ${safeName}-alias-filter-cont-body-s0-${k}`
      const canonicalCloseAlt = `ws ${invokeClose} turn-next-post`
      const aliasCloseAlt = `ws (${aliasInvokeCloseAlt}) turn-next-post`

      rules.set(
        `${safeName}-seq-${k}`,
        [canonicalParam, ...aliasCanonicalParams, canonicalFilterAlt, canonicalCloseAlt].join(' | '),
      )

      rules.set(
        `${safeName}-alias-seq-${k}`,
        [canonicalParamAlt(aliasBodyRule), ...aliasInvokeParams, aliasFilterAlt, aliasCloseAlt].join(' | '),
      )

      // Filter continuation: after filter close, chain back to same seq-K
      rules.set(
        `${safeName}-filter-cont-body-s0-${k}`,
        `filter-esc-body (("</magnitude:filter>" | "/magnitude:filter>") filter-greedy-body)* ("</magnitude:filter>" | "/magnitude:filter>") ${safeName}-seq-${k}`,
      )
      rules.set(
        `${safeName}-alias-filter-cont-body-s0-${k}`,
        `filter-esc-body (("</magnitude:filter>" | "/magnitude:filter>") filter-greedy-body)* ("</magnitude:filter>" | "/magnitude:filter>") ${safeName}-alias-seq-${k}`,
      )
    }

    // Non-last body rules: for each position K > 1, body chains to seq-(K-1)
    for (let k = N; k >= 2; k--) {
      const nextSeq = `${safeName}-seq-${k - 1}`
      const nextAliasSeq = `${safeName}-alias-seq-${k - 1}`

      const paramClose = invokeClose.replace('invoke', 'parameter')
      const paramSlashClose = `"/${invokeClose.replace('invoke', 'parameter').slice(2)}`  // "/magnitude:parameter>"
      const paramDualClose = `(${paramClose} | ${paramSlashClose})`

      rules.set(
        `${safeName}-nonlast-body-s0-${k}`,
        `param-esc-body (${paramDualClose} param-greedy-body)* ${paramDualClose} ${nextSeq}`,
      )
      rules.set(
        `${safeName}-alias-nonlast-body-s0-${k}`,
        `param-esc-body (${paramDualClose} param-greedy-body)* ${paramDualClose} ${nextAliasSeq}`,
      )

      for (const param of tool.parameters) {
        const pSafe = sanitizeParamName(param.name)
        const aliasEscBody = `${safeName}-${pSafe}-esc-body`
        const aliasGreedyBody = `${safeName}-${pSafe}-greedy-body`
        const aliasClose = `"</magnitude:${escapeGbnfString(param.name)}>"`
        const aliasSlashClose = `"/magnitude:${escapeGbnfString(param.name)}>"`
        const aliasDualClose = `(${aliasClose} | ${aliasSlashClose})`

        rules.set(
          `${safeName}-nonlast-body-s0-${k}-${pSafe}`,
          `${aliasEscBody} (${aliasDualClose} ${aliasGreedyBody})* ${aliasDualClose} ${nextSeq}`,
        )
        rules.set(
          `${safeName}-alias-nonlast-body-s0-${k}-${pSafe}`,
          `${aliasEscBody} (${aliasDualClose} ${aliasGreedyBody})* ${aliasDualClose} ${nextAliasSeq}`,
        )
      }

    }

    // Post-last-param: after the last parameter closes, accept filter or invoke close
    const postLastCanonical = `ws "<magnitude:filter>" ${safeName}-filter-cont-body-s0-1 | ws ${invokeClose} turn-next-post`
    const postLastAlias = `ws "<magnitude:filter>" ${safeName}-alias-filter-cont-body-s0-1 | ws (${aliasInvokeCloseAlt}) turn-next-post`
    rules.set(`${safeName}-post-last-param`, postLastCanonical)
    rules.set(`${safeName}-alias-post-last-param`, postLastAlias)

    const paramDualClose = `("</magnitude:parameter>" | "/magnitude:parameter>")`
    rules.set(
      `${safeName}-last-body-s0`,
      `param-esc-body (${paramDualClose} param-greedy-body)* ${paramDualClose} ${safeName}-post-last-param`,
    )
    rules.set(
      `${safeName}-alias-last-body-s0`,
      `param-esc-body (${paramDualClose} param-greedy-body)* ${paramDualClose} ${safeName}-alias-post-last-param`,
    )

    for (const param of tool.parameters) {
      const pSafe = sanitizeParamName(param.name)
      const aliasEscBody = `${safeName}-${pSafe}-esc-body`
      const aliasGreedyBody = `${safeName}-${pSafe}-greedy-body`
      const aliasClose = `"</magnitude:${escapeGbnfString(param.name)}>"`
      const aliasSlashClose = `"/magnitude:${escapeGbnfString(param.name)}>"`
      const aliasDualClose = `(${aliasClose} | ${aliasSlashClose})`

      rules.set(
        `${safeName}-last-body-s0-${pSafe}`,
        `${aliasEscBody} (${aliasDualClose} ${aliasGreedyBody})* ${aliasDualClose} ${safeName}-post-last-param`,
      )
      rules.set(
        `${safeName}-alias-last-body-s0-${pSafe}`,
        `${aliasEscBody} (${aliasDualClose} ${aliasGreedyBody})* ${aliasDualClose} ${safeName}-alias-post-last-param`,
      )
    }



    // Entry points
    rules.set(`${safeName}-body`, `${safeName}-seq-${N}`)
    rules.set(`${safeName}-alias-body`, `${safeName}-alias-seq-${N}`)
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

/**
 * Generate BUC rules that exclude BOTH a close tag AND an open tag.
 * Used to make body rules escape-aware: the body content excludes
 * its own close tag AND `<magnitude:escape>` so that escape blocks
 * can be recognized inline.
 *
 * Since close tags start with `</` and open tags start with `<` + letter,
 * after `<` the paths diverge and form independent prefix chains.
 */
export function generateCompoundBucRules(
  prefix: string,
  closeTagName: string,
  openTagName: string,
  options?: { plus?: boolean; excludeOpenPrefix?: boolean; excludeClosePrefix?: string; excludeChars?: string[]; afterLt?: boolean },
): string[] {
  const rules: string[] = []
  // State-machine BUC: after '<', enter chain check sub-rules.
  // If chain matches full excluded prefix → sub-rule fails → '<' not consumed → BUC stops.
  // If chain diverges → diverging char consumed → '<' + partial prefix + diverging char consumed as content.
  //
  // afterLt mode: BUC is used after a body-level '<' recovery. It additionally stops at
  // the close tag without the leading '<' (e.g., '/magnitude:parameter>'). This is needed
  // so that after consuming a lone '<' in content, the BUC correctly stops at the close tag.
  const closeSeq = options?.excludeClosePrefix
    ? '</' + options.excludeClosePrefix
    : '</' + closeTagName + '>'
  const openSeq = options?.excludeOpenPrefix ? '<' + openTagName : '<' + openTagName + '>'
  const extraExclude = (options?.excludeChars ?? []).map(escapeGbnfCharClass).join('')

  const closeChars = closeSeq.split('')
  const openChars = openSeq.split('')
  const closeBranch = closeChars[1] // '/'
  const openBranch = openChars[1]   // e.g., 'm' for 'magnitude:'

  // Build a chain of sub-rules for a character sequence (after the branch char).
  // At each position: if char matches, continue chain. If not, consume diverging char (success).
  // If entire chain matched, no production for the matching char → rule fails.
  function buildChain(chainId: string, chars: string[]): string {
    if (chars.length === 0) return '' // sentinel: chain completed

    const ch = chars[0]
    const chEsc = escapeGbnfCharClass(ch)
    const rest = chars.slice(1)
    const ruleId = `${chainId}-${chars.length}`

    if (rest.length === 0) {
      // Last char. Match → chain complete → fail. No match → consume diverging char.
      rules.push(`${ruleId} ::= [^${chEsc}]`)
      return ruleId
    }

    const nextRuleId = buildChain(chainId, rest)
    if (nextRuleId === '') {
      // Next step completes chain. Current char match → complete → fail.
      rules.push(`${ruleId} ::= [^${chEsc}]`)
    } else {
      rules.push(`${ruleId} ::= "${escapeGbnfString(ch)}" ${nextRuleId} | [^${chEsc}]`)
    }
    return ruleId
  }

  const closeChainId = buildChain(`${prefix}-cc`, closeChars.slice(2)) // chars after '</'
  const openChainId = buildChain(`${prefix}-oc`, openChars.slice(2))   // chars after '<m' etc.

  // After '<': branch on first char after '<'
  const ltAlts: string[] = []
  const closeBranchEsc = escapeGbnfCharClass(closeBranch)
  const openBranchEsc = escapeGbnfCharClass(openBranch)

  // Close branch
  if (closeChainId !== '') {
    ltAlts.push(`"${escapeGbnfString(closeBranch)}" ${closeChainId}`)
  }
  // else: close prefix is just '</' — seeing '/' means prefix complete → no alt

  // Open branch (only if different from close branch)
  if (closeBranch !== openBranch) {
    if (openChainId !== '') {
      ltAlts.push(`"${escapeGbnfString(openBranch)}" ${openChainId}`)
    }
    // else: open prefix is just '<m' — seeing 'm' means prefix complete → no alt
  }

  // Catch-all: '<' followed by char that's neither branch char nor '<'
  // Excluding '<' ensures '<<' doesn't consume both chars — the BUC stops,
  // letting the body-level '<' recovery handle lone '<' in content.
  const catchAllExclude = closeBranch === openBranch
    ? `${closeBranchEsc}<`
    : `${closeBranchEsc}${openBranchEsc}<`
  ltAlts.push(`[^${catchAllExclude}]`)

  const ltRuleId = `${prefix}-lt`
  rules.push(`${ltRuleId} ::= ${ltAlts.join(' | ')}`)

  // afterLt mode: also stop at the close tag without leading '<'
  // e.g., for close '</magnitude:parameter>', also stop at '/magnitude:parameter>'
  // This adds '/' as a top-level branch in the BUC (not just inside the lt sub-rule)
  if (options?.afterLt) {
    // Build chain for close tag without '</' — just 'magnitude:parameter>'
    const slashCloseSeq = options?.excludeClosePrefix
      ? '/' + options.excludeClosePrefix
      : '/' + closeTagName + '>'
    const slashCloseChars = slashCloseSeq.split('')
    // Chain starts after '/' — check 'magnitude:parameter>'
    const slashCloseChainId = buildChain(`${prefix}-sc`, slashCloseChars.slice(1))

    // Also stop at open prefix without '<' (e.g., 'magnitude:' when excludeOpenPrefix)
    // Build chain for open tag without '<' — e.g., 'magnitude:'
    const openWithoutLtSeq = options?.excludeOpenPrefix ? openTagName : openTagName + '>'
    const openWithoutLtChars = openWithoutLtSeq.split('')
    const openWithoutLtBranch = openWithoutLtChars[0] // 'm' for 'magnitude:'
    const openWithoutLtChainId = buildChain(`${prefix}-oc2`, openWithoutLtChars.slice(1))

    // Collect all top-level excluded chars for the first alt
    const topExclude = [escapeGbnfCharClass('<'), escapeGbnfCharClass('/')]
    if (openWithoutLtBranch) topExclude.push(escapeGbnfCharClass(openWithoutLtBranch))
    topExclude.push(extraExclude)

    const unitAlts = [
      `[^${topExclude.join('')}]`,
      `"<" ${ltRuleId}`,
    ]
    // '/' branch: check if it's the close tag without '<'
    if (slashCloseChainId !== '') {
      unitAlts.push(`"/" ${slashCloseChainId}`)
    }
    // '/' followed by non-close-tag char — consume as content
    const slashNextChar = slashCloseChars[1]
    if (slashNextChar) {
      unitAlts.push(`"/" [^${escapeGbnfCharClass(slashNextChar)}]`)
    }
    // 'm' branch (open prefix without '<'): check if it's the open prefix
    if (openWithoutLtChainId !== '') {
      unitAlts.push(`"${escapeGbnfString(openWithoutLtBranch)}" ${openWithoutLtChainId}`)
    }
    // 'm' followed by non-open-prefix char — consume as content
    const openNextChar = openWithoutLtChars[1]
    if (openNextChar) {
      unitAlts.push(`"${escapeGbnfString(openWithoutLtBranch)}" [^${escapeGbnfCharClass(openNextChar)}]`)
    }

    const quantifier = options?.plus ? '+' : '*'
    rules.push(`${prefix} ::= (${unitAlts.join(' | ')})${quantifier}`)
  } else {
    // Standard BUC: non-'<' chars OR '<' followed by chain check
    const unitAlts = [
      `[^${escapeGbnfCharClass('<')}${extraExclude}]`,
      `"<" ${ltRuleId}`,
    ]

    const quantifier = options?.plus ? '+' : '*'
    rules.push(`${prefix} ::= (${unitAlts.join(' | ')})${quantifier}`)
  }
  return rules
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
    case '\n': return '\\n'
    case '\t': return '\\t'
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
