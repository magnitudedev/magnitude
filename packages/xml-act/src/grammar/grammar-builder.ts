
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
  readonly required: boolean
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
    for (const child of proseChildren) {
      if (child === 'magnitude:reason') continue
      if (child === 'magnitude:message' && allowMessages) {
        postItems.push('"<magnitude:message" msg-attrs-opt ">" msg-body-s0')
      } else if (child === 'magnitude:invoke' && allowTools) {
        postItems.push('"<magnitude:invoke" invoke-attrs ">" invoke-body')
      }
    }

    const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"<magnitude:message" msg-attrs-opt ">" msg-body-s0'

    rules.set('turn-item-post', postItemRule)
    rules.set('turn-next-post', 'ws turn-item-post | ws yield')

    // Lens phase: reason + post-lens items
    const hasReason = (proseChildren as readonly string[]).includes('magnitude:reason')
    const lensItems = hasReason
      ? ['"<magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItems]
      : postItems

    rules.set('turn-item-lens', lensItems.join(' | '))
    rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
  }

  /**
   * Shared BUC (body-until-close) rules for each close tag name.
   * These are reused across all body rules for the same tag.
   */
  private addSharedBucRules(rules: RuleMap): void {
    const magnitudePrefix = 'magnitude:'

    for (const rule of generateBucRules('reason-buc', 'magnitude:reason')) {
      addRule(rules, rule)
    }

    for (const rule of generateCompoundBucRules('reason-body', 'magnitude:reason', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:' })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('msg-body', 'magnitude:message', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:' })) {
      addRule(rules, rule)
    }
    // param-body and filter-body: BUC stops at '<' that looks like close/open prefix.
    // To handle bare '<' in content (e.g., '<</magnitude:parameter>'), we use a
    // '<' recovery pattern: buc ("<" alt-buc)* — the outer rule consumes '<' and
    // enters an afterLt BUC that checks for '/magnitude:' without the leading '<'.
    for (const rule of generateCompoundBucRules('param-body-buc', 'magnitude:parameter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:' })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('param-body-alt', 'magnitude:parameter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', afterLt: true })) {
      addRule(rules, rule)
    }
    rules.set('param-body', 'param-body-buc ("<" param-body-alt)*')

    for (const rule of generateCompoundBucRules('filter-body-buc', 'magnitude:filter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:' })) {
      addRule(rules, rule)
    }
    for (const rule of generateCompoundBucRules('filter-body-alt', 'magnitude:filter', magnitudePrefix, { excludeOpenPrefix: true, excludeClosePrefix: 'magnitude:', afterLt: true })) {
      addRule(rules, rule)
    }
    rules.set('filter-body', 'filter-body-buc ("<" filter-body-alt)*')
  }

  /**
   * Top-level body rules using recursive greedy last-match.
   * Confirmation: </tagname> + ws + < (next structural tag).
   */
  private addTopLevelBodyRules(rules: RuleMap): void {
    rules.set('reason-body-s0', 'reason-body "</magnitude:reason>" turn-next-lens')
    rules.set('msg-body-s0', 'msg-body "</magnitude:message>" turn-next-post')
  }

  /**
   * Per-tool grammar rules with constrained param names, bounded counts,
   * and position-aware greedy matching.
   */
  private addToolRules(rules: RuleMap): void {
    const tools = this.config.tools

    if (tools.length === 0) {
      rules.set('invoke-attrs', '" tool=\\"" quoted-value "\\""')
      rules.set('invoke-body', 'ws invoke-generic-item | ws "</magnitude:invoke>" turn-next-post')
      rules.set('invoke-generic-item',
        '"<magnitude:parameter" " name=\\"" quoted-value "\\"" ">" generic-param-body-s0 | "<magnitude:filter>" generic-filter-body-s0')
      rules.set('generic-param-body-s0',
        'param-body "</magnitude:parameter>" (ws invoke-generic-item | ws "</magnitude:invoke>" turn-next-post)')
      rules.set('generic-filter-body-s0',
        'filter-body "</magnitude:filter>" (ws invoke-generic-item | ws "</magnitude:invoke>" turn-next-post)')
      return
    }

    const toolNameAlts = tools.map(t => `" tool=\\"${escapeGbnfString(t.tagName)}\\""`)
    rules.set('invoke-attrs', toolNameAlts.join(' | '))

    const invokeAlts: string[] = []

    for (const tool of tools) {
      const safeName = sanitizeRuleName(tool.tagName)

      this.addPerToolRules(rules, tool, safeName)

      invokeAlts.push(`"<magnitude:invoke" " tool=\\"${escapeGbnfString(tool.tagName)}\\"" ">" ${safeName}-body`)
    }

    const { allowMessages } = this.config.protocol
    const proseChildren = VALID_CHILDREN.prose

    const postItems: string[] = []
    for (const child of proseChildren) {
      if (child === 'magnitude:reason') continue
      if (child === 'magnitude:message' && allowMessages) {
        postItems.push('"<magnitude:message" msg-attrs-opt ">" msg-body-s0')
      } else if (child === 'magnitude:invoke') {
        postItems.push(...invokeAlts)
      }
    }

    const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"<magnitude:message" msg-attrs-opt ">" msg-body-s0'

    rules.set('turn-item-post', postItemRule)
    rules.set('turn-next-post', 'ws turn-item-post | ws yield')

    const hasReason = (proseChildren as readonly string[]).includes('magnitude:reason')
    const lensItems = hasReason
      ? ['"<magnitude:reason" reason-attrs-opt ">" reason-body-s0', ...postItems]
      : postItems

    rules.set('turn-item-lens', lensItems.join(' | '))
    rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
  }

  /**
   * Generate per-tool rules: param name constraints, bounded count,
   * position-aware greedy body rules.
   */
  private addPerToolRules(rules: RuleMap, tool: GrammarToolDef, safeName: string): void {
    const N = tool.parameters.length
    const requiredCount = tool.parameters.filter(p => p.required).length
    const invokeClose = `${safeName}-invoke-close`

    const canonicalParamBodyRule = (k: number): string =>
      k === 1 ? `${safeName}-last-body-s0` : `${safeName}-nonlast-body-s0-${k}`

    const canonicalParamAlt = (bodyRule: string): string =>
      `ws "<magnitude:parameter" ${safeName}-param-names ">" ${bodyRule}`

    const paramNameAlts = tool.parameters.map(p =>
      `" name=\\"${escapeGbnfString(p.name)}\\""`)
    if (paramNameAlts.length > 0) {
      rules.set(`${safeName}-param-names`, paramNameAlts.join(' | '))
    }

    const paramCloseAlts = ['"</magnitude:parameter>"', ...tool.parameters.map(
      p => `"</magnitude:${escapeGbnfString(p.name)}>"`,
    )]
    rules.set(`${safeName}-param-close`, paramCloseAlts.join(' | '))
    rules.set(`${safeName}-invoke-close`, `"</magnitude:invoke>" | "</magnitude:${escapeGbnfString(tool.tagName)}>"`)

    if (N === 0) {
      rules.set(`${safeName}-body`, `ws ${invokeClose} turn-next-post`)
      return
    }

    for (let k = N; k >= 1; k--) {
      const consumed = N - k
      const closeAllowed = consumed >= requiredCount
      const canonicalBodyRule = canonicalParamBodyRule(k)
      const canonicalParam = canonicalParamAlt(canonicalBodyRule)
      const canonicalFilterAlt = `ws "<magnitude:filter>" ${safeName}-filter-cont-body-s0-${k}`
      const canonicalCloseAlt = `ws ${invokeClose} turn-next-post`

      rules.set(
        `${safeName}-seq-${k}`,
        [canonicalParam, canonicalFilterAlt, ...(closeAllowed ? [canonicalCloseAlt] : [])].join(' | '),
      )

      rules.set(
        `${safeName}-filter-cont-body-s0-${k}`,
        `filter-body "</magnitude:filter>" ${safeName}-seq-${k}`,
      )
    }

    for (let k = N; k >= 2; k--) {
      const nextSeq = `${safeName}-seq-${k - 1}`

      rules.set(
        `${safeName}-nonlast-body-s0-${k}`,
        `param-body ${safeName}-param-close ${nextSeq}`,
      )
    }

    const postLastCanonical = `ws "<magnitude:filter>" ${safeName}-filter-cont-body-s0-1 | ws ${invokeClose} turn-next-post`
    rules.set(`${safeName}-post-last-param`, postLastCanonical)

    rules.set(
      `${safeName}-last-body-s0`,
      `param-body ${safeName}-param-close ${safeName}-post-last-param`,
    )

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

    if (maxLenses !== undefined) {
      for (let k = maxLenses; k >= 0; k--) {
        if (k === 0) {
          rules.set(`turn-next-forced-0`, 'ws forced-msg')
        } else {
          const nextK = k - 1
          rules.set(`reason-forced-${k}-body-s0`,
            `reason-buc "</magnitude:reason>" turn-next-forced-${nextK}`)
          rules.set(
            `turn-next-forced-${k}`,
            `ws "<magnitude:reason" reason-attrs-opt ">" reason-forced-${k}-body-s0 | ws forced-msg`
          )
        }
      }
      rules.set('root', `turn-next-forced-${maxLenses}`)
    } else {
      rules.set('reason-forced-body-s0',
        'reason-buc "</magnitude:reason>" turn-next-forced')
      rules.set(
        'turn-next-forced',
        'ws "<magnitude:reason" reason-attrs-opt ">" reason-forced-body-s0 | ws forced-msg'
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
 * Generate BUC rules that exclude BOTH a close tag AND an open tag prefix.
 * The body content stops at both the close tag prefix and the open tag prefix,
 * allowing structural tags to be recognized at body boundaries.
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


