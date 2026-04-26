import type { GrammarConfig, GrammarToolDef } from './grammar-types'
import {
  TAG_THINK,
  TAG_THINK_CLOSE_ALIAS,
  TAG_MESSAGE,
  TAG_INVOKE,
  TAG_PARAMETER,
  TAG_FILTER,
  MAGNITUDE_PREFIX,
  escapeGbnfString,
  sanitizeRuleName,
  gbnfCloseTag,
} from './grammar-utils'
import { generateCompoundBucRules } from './buc-rules'
import { VALID_CHILDREN } from '../nesting'

type RuleMap = Map<string, string>

function addRule(rules: RuleMap, line: string): void {
  const match = line.match(/^(\S+) ::= (.+)$/)
  if (match) rules.set(match[1], match[2])
}

// =============================================================================
// Whitespace & Attributes
// =============================================================================

export function addWhitespaceRules(rules: RuleMap): void {
  rules.set('ws', '[ \\t\\n]*')
}

export function addAttributeRules(rules: RuleMap): void {
  rules.set('quoted-value', '[^"]*')
  rules.set('think-attrs', '" about=\\"" quoted-value "\\""')
  rules.set('think-attrs-opt', 'think-attrs | ""')
  rules.set('msg-attrs', '" to=\\"" quoted-value "\\""')
  rules.set('msg-attrs-opt', 'msg-attrs | ""')
}

// =============================================================================
// Yield
// =============================================================================

export function addYieldRules(rules: RuleMap, yieldTags: ReadonlyArray<string>): void {
  const withLt = yieldTags.map(t => '"<' + t + '/>"')
  const noLt = yieldTags.map(t => '"' + t + '/>"')
  rules.set('yield', withLt.join(' | '))
  rules.set('yield-no-lt', noLt.join(' | '))
}

// =============================================================================
// Continuation
// =============================================================================

export function addContinuationRules(rules: RuleMap, config: GrammarConfig): void {
  const { allowMessages, allowTools } = config.protocol
  const proseChildren = VALID_CHILDREN.prose

  // Post-lens phase: message and/or invoke, then yield
  const postItems: string[] = []
  for (const child of proseChildren) {
    if (child === TAG_THINK) continue
    if (child === TAG_MESSAGE && allowMessages) {
      postItems.push('"' + '<' + TAG_MESSAGE + '"' + ' msg-attrs-opt ">" msg-body-s0')
    } else if (child === TAG_INVOKE && allowTools) {
      postItems.push('"' + '<' + TAG_INVOKE + '"' + ' invoke-attrs ">" invoke-body')
    }
  }

  const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"' + '<' + TAG_MESSAGE + '"' + ' msg-attrs-opt ">" msg-body-s0'

  rules.set('turn-item-post', postItemRule)
  rules.set('turn-next-post', 'ws turn-item-post | ws yield')

  // Lens phase: think + post-lens items
  const hasThink = (proseChildren as readonly string[]).includes(TAG_THINK)
  const lensItems = hasThink
    ? ['"' + '<' + TAG_THINK + '"' + ' think-attrs-opt ">" think-body-s0', ...postItems]
    : postItems

  rules.set('turn-item-lens', lensItems.join(' | '))
  rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
}

// =============================================================================
// Shared BUC
// =============================================================================

export function addSharedBucRules(rules: RuleMap): void {
  for (const rule of generateCompoundBucRules('think-buc', TAG_THINK, MAGNITUDE_PREFIX, { excludeOpenPrefix: true, excludeClosePrefix: MAGNITUDE_PREFIX })) {
    addRule(rules, rule)
  }

  for (const rule of generateCompoundBucRules('think-body', TAG_THINK, MAGNITUDE_PREFIX, { excludeOpenPrefix: true, excludeClosePrefix: MAGNITUDE_PREFIX })) {
    addRule(rules, rule)
  }

  // Alias close BUC: stops at the alias close tag prefix only
  for (const rule of generateCompoundBucRules('think-body-alias', TAG_THINK_CLOSE_ALIAS, MAGNITUDE_PREFIX, { excludeOpenPrefix: true })) {
    addRule(rules, rule)
  }

  for (const rule of generateCompoundBucRules('msg-body', TAG_MESSAGE, MAGNITUDE_PREFIX, { excludeOpenPrefix: true, excludeClosePrefix: MAGNITUDE_PREFIX })) {
    addRule(rules, rule)
  }

  // param-body and filter-body: BUC stops at '<' that looks like close/open prefix.
  // To handle bare '<' in content, we use a '<' recovery pattern:
  // buc ("<" alt-buc)* — the outer rule consumes '<' and enters an afterLt BUC.
  for (const rule of generateCompoundBucRules('param-body-buc', TAG_PARAMETER, MAGNITUDE_PREFIX, { excludeOpenPrefix: true, excludeClosePrefix: MAGNITUDE_PREFIX })) {
    addRule(rules, rule)
  }
  for (const rule of generateCompoundBucRules('param-body-alt', TAG_PARAMETER, MAGNITUDE_PREFIX, { excludeOpenPrefix: true, excludeClosePrefix: MAGNITUDE_PREFIX, afterLt: true })) {
    addRule(rules, rule)
  }
  rules.set('param-body', 'param-body-buc ("<" param-body-alt)*')

  for (const rule of generateCompoundBucRules('filter-body-buc', TAG_FILTER, MAGNITUDE_PREFIX, { excludeOpenPrefix: true, excludeClosePrefix: MAGNITUDE_PREFIX })) {
    addRule(rules, rule)
  }
  for (const rule of generateCompoundBucRules('filter-body-alt', TAG_FILTER, MAGNITUDE_PREFIX, { excludeOpenPrefix: true, excludeClosePrefix: MAGNITUDE_PREFIX, afterLt: true })) {
    addRule(rules, rule)
  }
  rules.set('filter-body', 'filter-body-buc ("<" filter-body-alt)*')
}

// =============================================================================
// Top-level body
// =============================================================================

export function addTopLevelBodyRules(rules: RuleMap): void {
  rules.set('think-body-s0', 'think-body ' + gbnfCloseTag(TAG_THINK) + ' turn-next-lens | think-body-alias ' + gbnfCloseTag(TAG_THINK_CLOSE_ALIAS) + ' turn-next-lens')
  rules.set('msg-body-s0', 'msg-body ' + gbnfCloseTag(TAG_MESSAGE) + ' turn-next-post')
}

// =============================================================================
// Per-tool rules
// =============================================================================

export function addToolRules(rules: RuleMap, config: GrammarConfig): void {
  const tools = config.tools

  if (tools.length === 0) {
    rules.set('invoke-attrs', '" tool=\\"" quoted-value "\\""')
    rules.set('invoke-body', 'ws invoke-generic-item | ws ' + gbnfCloseTag(TAG_INVOKE) + ' turn-next-post')
    rules.set('invoke-generic-item',
      '"' + '<' + TAG_PARAMETER + '"' + ' " name=\\"" quoted-value "\\"" ">" generic-param-body-s0 | "' + '<' + TAG_FILTER + '>" generic-filter-body-s0')
    rules.set('generic-param-body-s0',
      'param-body ' + gbnfCloseTag(TAG_PARAMETER) + ' (ws invoke-generic-item | ws ' + gbnfCloseTag(TAG_INVOKE) + ' turn-next-post)')
    rules.set('generic-filter-body-s0',
      'filter-body ' + gbnfCloseTag(TAG_FILTER) + ' (ws invoke-generic-item | ws ' + gbnfCloseTag(TAG_INVOKE) + ' turn-next-post)')
    return
  }

  const toolNameAlts = tools.map(t => '" tool=\\"' + escapeGbnfString(t.tagName) + '\\""')
  rules.set('invoke-attrs', toolNameAlts.join(' | '))

  const invokeAlts: string[] = []

  for (const tool of tools) {
    const safeName = sanitizeRuleName(tool.tagName)

    addPerToolRules(rules, tool, safeName)

    invokeAlts.push('"' + '<' + TAG_INVOKE + '" " tool=\\"' + escapeGbnfString(tool.tagName) + '\\"" ">" ' + safeName + '-body')
  }

  const { allowMessages } = config.protocol
  const proseChildren = VALID_CHILDREN.prose

  const postItems: string[] = []
  for (const child of proseChildren) {
    if (child === TAG_THINK) continue
    if (child === TAG_MESSAGE && allowMessages) {
      postItems.push('"' + '<' + TAG_MESSAGE + '"' + ' msg-attrs-opt ">" msg-body-s0')
    } else if (child === TAG_INVOKE) {
      postItems.push(...invokeAlts)
    }
  }

  const postItemRule = postItems.length > 0 ? postItems.join(' | ') : '"' + '<' + TAG_MESSAGE + '"' + ' msg-attrs-opt ">" msg-body-s0'

  rules.set('turn-item-post', postItemRule)
  rules.set('turn-next-post', 'ws turn-item-post | ws yield')

  const hasThink = (proseChildren as readonly string[]).includes(TAG_THINK)
  const lensItems = hasThink
    ? ['"' + '<' + TAG_THINK + '"' + ' think-attrs-opt ">" think-body-s0', ...postItems]
    : postItems

  rules.set('turn-item-lens', lensItems.join(' | '))
  rules.set('turn-next-lens', 'ws turn-item-lens | ws yield')
}

function addPerToolRules(rules: RuleMap, tool: GrammarToolDef, safeName: string): void {
  const N = tool.parameters.length
  const requiredCount = tool.parameters.filter(p => p.required).length
  const invokeClose = `${safeName}-invoke-close`

  const canonicalParamBodyRule = (k: number): string =>
    k === 1 ? `${safeName}-last-body-s0` : `${safeName}-nonlast-body-s0-${k}`

  const canonicalParamAlt = (bodyRule: string): string =>
    'ws "' + '<' + TAG_PARAMETER + '" ' + safeName + '-param-names ">" ' + bodyRule

  const paramNameAlts = tool.parameters.map(p =>
    '" name=\\"' + escapeGbnfString(p.name) + '\\""')
  if (paramNameAlts.length > 0) {
    rules.set(`${safeName}-param-names`, paramNameAlts.join(' | '))
  }

  const paramCloseAlts = [gbnfCloseTag(TAG_PARAMETER), ...tool.parameters.map(
    p => gbnfCloseTag(MAGNITUDE_PREFIX + p.name),
  )]
  rules.set(`${safeName}-param-close`, paramCloseAlts.join(' | '))
  rules.set(`${safeName}-invoke-close`, gbnfCloseTag(TAG_INVOKE) + ' | ' + gbnfCloseTag(MAGNITUDE_PREFIX + tool.tagName))

  if (N === 0) {
    rules.set(`${safeName}-body`, `ws ${invokeClose} turn-next-post`)
    return
  }

  for (let k = N; k >= 1; k--) {
    const consumed = N - k
    const closeAllowed = consumed >= requiredCount
    const canonicalBodyRule = canonicalParamBodyRule(k)
    const canonicalParam = canonicalParamAlt(canonicalBodyRule)
    const canonicalFilterAlt = 'ws "' + '<' + TAG_FILTER + '>" ' + safeName + '-filter-cont-body-s0-' + k
    const canonicalCloseAlt = `ws ${invokeClose} turn-next-post`

    rules.set(
      `${safeName}-seq-${k}`,
      [canonicalParam, canonicalFilterAlt, ...(closeAllowed ? [canonicalCloseAlt] : [])].join(' | '),
    )

    rules.set(
      `${safeName}-filter-cont-body-s0-${k}`,
      'filter-body ' + gbnfCloseTag(TAG_FILTER) + ' ' + safeName + '-seq-' + k,
    )
  }

  for (let k = N; k >= 2; k--) {
    const nextSeq = `${safeName}-seq-${k - 1}`

    rules.set(
      `${safeName}-nonlast-body-s0-${k}`,
      'param-body ' + safeName + '-param-close ' + nextSeq,
    )
  }

  const postLastCanonical = 'ws "' + '<' + TAG_FILTER + '>" ' + safeName + '-filter-cont-body-s0-1 | ws ' + invokeClose + ' turn-next-post'
  rules.set(`${safeName}-post-last-param`, postLastCanonical)

  rules.set(
    `${safeName}-last-body-s0`,
    'param-body ' + safeName + '-param-close ' + safeName + '-post-last-param',
  )

  rules.set(`${safeName}-body`, `${safeName}-seq-${N}`)
}

// =============================================================================
// Root & Forced Message
// =============================================================================

export function addRootRule(rules: RuleMap, config: GrammarConfig): void {
  const { minLenses, requiredMessageTo, maxLenses } = config.protocol

  if (requiredMessageTo !== null) {
    addForcedMessageRules(rules, requiredMessageTo, maxLenses)
  } else if (minLenses === 1) {
    rules.set('root', 'ws "' + '<' + TAG_THINK + '"' + ' think-attrs-opt ">" think-body-s0')
  } else {
    rules.set('root', 'turn-next-lens')
  }
}

function addForcedMessageRules(rules: RuleMap, recipient: string, maxLenses: number | undefined): void {
  const escapedRecipient = recipient.replace(/"/g, '\\"')
  rules.set('forced-msg', '"' + '<' + TAG_MESSAGE + ' to=\\"' + escapedRecipient + '\\">" msg-body-s0')

  if (maxLenses !== undefined) {
    for (let k = maxLenses; k >= 0; k--) {
      if (k === 0) {
        rules.set(`turn-next-forced-0`, 'ws forced-msg')
      } else {
        const nextK = k - 1
        rules.set(`think-forced-${k}-body-s0`,
          'think-buc ' + gbnfCloseTag(TAG_THINK) + ' turn-next-forced-' + nextK + ' | think-buc ' + gbnfCloseTag(TAG_THINK_CLOSE_ALIAS) + ' turn-next-forced-' + nextK)
        rules.set(
          `turn-next-forced-${k}`,
          'ws "' + '<' + TAG_THINK + '"' + ' think-attrs-opt ">" think-forced-' + k + '-body-s0 | ws forced-msg'
        )
      }
    }
    rules.set('root', `turn-next-forced-${maxLenses}`)
  } else {
    rules.set('think-forced-body-s0',
      'think-buc ' + gbnfCloseTag(TAG_THINK) + ' turn-next-forced | think-buc ' + gbnfCloseTag(TAG_THINK_CLOSE_ALIAS) + ' turn-next-forced')
    rules.set(
      'turn-next-forced',
      'ws "' + '<' + TAG_THINK + '"' + ' think-attrs-opt ">" think-forced-body-s0 | ws forced-msg'
    )
    rules.set('root', 'turn-next-forced')
  }
}
