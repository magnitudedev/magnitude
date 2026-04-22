
import { LEAD_YIELD_TAGS } from '../constants'
import { VALID_CHILDREN } from '../nesting'

// =============================================================================
// Types
// =============================================================================

/**
 * Parameter binding for grammar generation.
 * Kept for API compatibility — tool parameters are no longer enumerated in the grammar.
 */
export interface GrammarParameterDef {
  readonly name: string
  readonly field: string
  readonly type: 'scalar' | 'json'
}

/**
 * Tool definition for grammar generation.
 * Kept for API compatibility — tools are no longer enumerated in the grammar.
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

/**
 * Context for body DFA generation — specifies which continuation rules
 * the trailing-whitespace states should reference when the close tag is confirmed.
 */
export interface BodyContext {
  /** Rule to invoke after "\n" confirms the close tag. */
  readonly confirmRule: string
  /** Rule to invoke after "<" confirms the close tag (< already consumed). */
  readonly confirmNoLtRule: string
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
    this.addInvokeInternalContinuationRules(rules)
    this.addContinuationRules(rules)
    this.addSharedBodyRules(rules)
    this.addRootRule(rules)

    return serializeGrammar(rules)
  }

  // ---------------------------------------------------------------------------
  // Rule contributors
  // ---------------------------------------------------------------------------

  private addWhitespaceRules(rules: RuleMap): void {
    // ws: unbounded whitespace — used before block elements
    rules.set('ws', '[ \\t\\n]*')
  }

  private addAttributeRules(rules: RuleMap): void {
    rules.set('quoted-value', '[^"]*')
    rules.set('reason-attrs', '" about=\\"" quoted-value "\\""')
    rules.set('reason-attrs-opt', 'reason-attrs | ""')
    rules.set('msg-attrs', '" to=\\"" quoted-value "\\""')
    rules.set('invoke-attrs', '" tool=\\"" quoted-value "\\""')
    rules.set('param-attrs', '" name=\\"" quoted-value "\\""')
  }

  private addYieldRules(rules: RuleMap): void {
    const yieldTags = this.config.protocol.yieldTags
    const withLt = yieldTags.map(t => `"<${t}/>"`)
    const noLt = yieldTags.map(t => `"${t}/>"`)
    rules.set('yield', withLt.join(' | '))
    rules.set('yield-no-lt', noLt.join(' | '))
  }

  private addInvokeInternalContinuationRules(rules: RuleMap): void {
    // Derive invoke children from VALID_CHILDREN — provably in sync with the parser
    const invokeChildren = VALID_CHILDREN.invoke  // ['parameter', 'filter']

    const invokeItemAlts: string[] = []
    const invokeItemNoLtAlts: string[] = []
    for (const child of invokeChildren) {
      if (child === 'parameter') {
        invokeItemAlts.push('"<parameter" param-attrs ">" param-body-s0')
        invokeItemNoLtAlts.push('"parameter" param-attrs ">" param-body-s0')
      } else if (child === 'filter') {
        invokeItemAlts.push('"<filter>" filter-body-s0')
        invokeItemNoLtAlts.push('"filter>" filter-body-s0')
      }
    }

    rules.set('invoke-item', invokeItemAlts.join(' | '))
    rules.set('invoke-next', 'ws invoke-item | ws "</invoke>" turn-next-post')
    rules.set(
      'invoke-next-no-lt',
      [...invokeItemNoLtAlts, '"/invoke>" turn-next-post'].join(' | ')
    )
  }

  private addContinuationRules(rules: RuleMap): void {
    const { allowMessages, allowTools } = this.config.protocol

    // Derive post-lens children from VALID_CHILDREN.prose (excludes 'reason' which is lens-phase only)
    // VALID_CHILDREN.prose = ['reason', 'message', 'invoke']
    const proseChildren = VALID_CHILDREN.prose

    // Post-lens phase: message and/or invoke, then yield
    const postItems: string[] = []
    const postItemsNoLt: string[] = []
    for (const child of proseChildren) {
      if (child === 'reason') continue  // reason is lens-phase only
      if (child === 'message' && allowMessages) {
        postItems.push('"<message" msg-attrs ">" msg-body-s0')
        postItemsNoLt.push('"message" msg-attrs ">" msg-body-s0')
      } else if (child === 'invoke' && allowTools) {
        postItems.push('"<invoke" invoke-attrs ">" invoke-next')
        postItemsNoLt.push('"invoke" invoke-attrs ">" invoke-next')
      }
    }

    const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"<message" msg-attrs ">" msg-body-s0'
    const postNoLtItems = postItemsNoLt.length > 0 ? postItemsNoLt : ['"message" msg-attrs ">" msg-body-s0']

    rules.set('turn-item-post', postItemRule)
    rules.set('turn-next-post', 'ws turn-item-post | ws yield')
    rules.set('turn-next-post-no-lt', [...postNoLtItems, 'yield-no-lt'].join(' | '))

    // Lens phase: reason (from VALID_CHILDREN.prose) + post-lens items
    const hasReason = (proseChildren as readonly string[]).includes('reason')
    const lensItems = hasReason
      ? ['"<reason" reason-attrs-opt ">" reason-body-s0', ...postItems]
      : postItems
    const lensItemsNoLt = hasReason
      ? ['"reason" reason-attrs-opt ">" reason-body-s0', ...postItemsNoLt]
      : postItemsNoLt

    rules.set('turn-item-lens', lensItems.join(' | '))
    rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
    rules.set('turn-next-lens-no-lt', [...lensItemsNoLt, 'yield-no-lt'].join(' | '))
  }

  private addSharedBodyRules(rules: RuleMap): void {
    const lensCtx: BodyContext = {
      confirmRule: 'turn-next-lens',
      confirmNoLtRule: 'turn-next-lens-no-lt',
    }
    const postCtx: BodyContext = {
      confirmRule: 'turn-next-post',
      confirmNoLtRule: 'turn-next-post-no-lt',
    }
    const invokeCtx: BodyContext = {
      confirmRule: 'invoke-next',
      confirmNoLtRule: 'invoke-next-no-lt',
    }

    for (const rule of generateBodyRules('reason-body', 'reason', lensCtx)) {
      addRule(rules, rule)
    }
    for (const rule of generateBodyRules('msg-body', 'message', postCtx)) {
      addRule(rules, rule)
    }
    for (const rule of generateBodyRules('param-body', 'parameter', invokeCtx)) {
      addRule(rules, rule)
    }
    for (const rule of generateBodyRules('filter-body', 'filter', invokeCtx)) {
      addRule(rules, rule)
    }
  }

  private addRootRule(rules: RuleMap): void {
    const { minLenses, requiredMessageTo, maxLenses } = this.config.protocol

    if (requiredMessageTo !== null) {
      this.addForcedMessageRules(rules, requiredMessageTo, maxLenses)
      // root set inside addForcedMessageRules
    } else if (minLenses === 1) {
      // Must start with at least one reason; reason body chains back to lens phase
      rules.set('root', 'ws "<reason" reason-attrs-opt ">" reason-body-s0')
    } else {
      rules.set('root', 'turn-next-lens')
    }
  }

  private addForcedMessageRules(rules: RuleMap, recipient: string, maxLenses: number | undefined): void {
    // Forced message rule (literal recipient)
    const escapedRecipient = recipient.replace(/"/g, '\\"')
    rules.set('forced-msg', `"<message to=\\"${escapedRecipient}\\">" msg-body-s0`)
    rules.set('forced-msg-no-lt', `"message to=\\"${escapedRecipient}\\">" msg-body-s0`)

    if (maxLenses !== undefined) {
      // Generate N+1 forced-phase variants, each allowing one fewer reason.
      // turn-next-forced-N allows N more reasons then forced-msg
      // turn-next-forced-0 allows only forced-msg
      for (let k = maxLenses; k >= 0; k--) {
        if (k === 0) {
          rules.set(`turn-next-forced-0`, 'ws forced-msg')
          rules.set(`turn-next-forced-0-no-lt`, 'forced-msg-no-lt')
        } else {
          const nextK = k - 1
          // reason body DFA for slot k chains to turn-next-forced-(k-1)
          const reasonCtx: BodyContext = {
            confirmRule: `turn-next-forced-${nextK}`,
            confirmNoLtRule: `turn-next-forced-${nextK}-no-lt`,
          }
          for (const rule of generateBodyRules(`reason-forced-${k}-body`, 'reason', reasonCtx)) {
            addRule(rules, rule)
          }
          rules.set(
            `turn-next-forced-${k}`,
            `ws "<reason" reason-attrs-opt ">" reason-forced-${k}-body-s0 | ws forced-msg`
          )
          rules.set(
            `turn-next-forced-${k}-no-lt`,
            `"reason" reason-attrs-opt ">" reason-forced-${k}-body-s0 | forced-msg-no-lt`
          )
        }
      }
      rules.set('root', `turn-next-forced-${maxLenses}`)
    } else {
      // No maxLenses: unlimited reasons before forced message
      // reason-forced-body chains back to turn-next-forced
      const reasonCtx: BodyContext = {
        confirmRule: 'turn-next-forced',
        confirmNoLtRule: 'turn-next-forced-no-lt',
      }
      for (const rule of generateBodyRules('reason-forced-body', 'reason', reasonCtx)) {
        addRule(rules, rule)
      }
      rules.set(
        'turn-next-forced',
        'ws "<reason" reason-attrs-opt ">" reason-forced-body-s0 | ws forced-msg'
      )
      rules.set(
        'turn-next-forced-no-lt',
        '"reason" reason-attrs-opt ">" reason-forced-body-s0 | forced-msg-no-lt'
      )
      rules.set('root', 'turn-next-forced')
    }
  }
}

// =============================================================================
// DFA Body Rule Generation
// =============================================================================

/**
 * Generate DFA body rules for a tag with the given close-tag name.
 *
 * The DFA matches any content, rejecting false close tags (ones not followed
 * by bounded whitespace then `\n` or a known next-tag prefix).
 *
 * When the close tag IS confirmed, the DFA hands off to the specified
 * continuation rules rather than terminating — enabling the chain grammar.
 *
 * @param prefix      Rule name prefix, e.g. "reason-body"
 * @param tagName     The close tag name, e.g. "reason"
 * @param context     Which continuation rules to invoke on confirmation
 */
export function generateBodyRules(prefix: string, tagName: string, context: BodyContext): string[] {
  const lines: string[] = []
  const L = tagName.length
  const MAX_TW = 4

  // Entry alias
  lines.push(`${prefix} ::= ${prefix}-s0`)

  // s0: base content state — consume non-'<' freely, enter close-tag matching on '<'
  lines.push(`${prefix}-s0 ::= [^<] ${prefix}-s0 | "<" ${prefix}-s1`)

  // s1: saw '<' — check for '/' (close tag) or restart on another '<'
  lines.push(`${prefix}-s1 ::= "/" ${prefix}-sl | "<" ${prefix}-s1 | [^/<] ${prefix}-s0`)

  // sl: saw '</' — match first char of tag name
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

    // gt: consumed full tag name, now consume '>'
    lines.push(`${prefix}-gt ::= ">" ${prefix}-tw0 | "<" ${prefix}-s1 | [^<>] ${prefix}-s0`)
  }

  // tw0..tw{MAX_TW}: trailing whitespace states after the '>'
  // The model is NEVER constrained here — every character has a valid transition.
  // Horizontal whitespace advances the window; \n or < confirms the close;
  // any other character (including excess whitespace at twMAX) rejects back to s0.
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
      // Final tw state: no more whitespace advancement.
      // [^\n<] (no space excluded) ensures excess spaces/tabs escape to s0 rather than dead-ending.
      lines.push(
        `${stateName} ::= "\\n" ${context.confirmRule}` +
        ` | "<" ${context.confirmNoLtRule}` +
        ` | [^ \\n<] ${prefix}-s0`
      )
    }
  }

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

export function sanitizeRuleName(tagName: string): string {
  return `t-${tagName.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()}`
}

export function sanitizeParamName(name: string): string {
  return name.replace(/[^a-zA-Z0-9]/g, '').toLowerCase()
}
