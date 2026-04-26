import { escapeGbnfCharClass, escapeGbnfString } from './grammar-utils'

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
  options?: { plus?: boolean; excludeOpenPrefix?: boolean; excludeClosePrefix?: string; excludeChars?: string[]; afterLt?: boolean; closeAliases?: string[] },
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

  // Build alias close chains (short-form close tags like 'think' for 'magnitude:think')
  const aliasChains: Array<{ firstChar: string; chainId: string }> = []
  for (const alias of (options?.closeAliases ?? [])) {
    const aliasCloseSeq = '</' + alias + '>'
    const aliasCloseChars = aliasCloseSeq.split('')
    // After '</', the first char is the first char of the alias name
    const aliasFirstChar = aliasCloseChars[2] // e.g., 't' for 'think'
    const aliasChainId = buildChain(`${prefix}-ac-${alias}`, aliasCloseChars.slice(3)) // chars after '</t'
    aliasChains.push({ firstChar: aliasFirstChar, chainId: aliasChainId })
  }

  // After '<': branch on first char after '<'
  const ltAlts: string[] = []
  const closeBranchEsc = escapeGbnfCharClass(closeBranch)
  const openBranchEsc = escapeGbnfCharClass(openBranch)

  // Close branch — handle primary and alias close tags
  if (closeChainId !== '' || aliasChains.length > 0) {
    if (aliasChains.length === 0) {
      // Simple case: no aliases, just the primary close chain
      ltAlts.push(`"${escapeGbnfString(closeBranch)}" ${closeChainId}`)
    } else {
      // Put close alternatives directly at lt level (not in a cs sub-rule)
      // to avoid gbnf pointer fan-out from sub-rule branching.
      // After '<': '/' + primaryFirstChar → primary close chain
      //            '/' + aliasFirstChar → alias close chain
      //            '/' + other → divergence (consume as body)
      if (closeChainId !== '') {
        const primaryFirstChar = closeChars[2]
        ltAlts.push(`"${escapeGbnfString(closeBranch)}${escapeGbnfString(primaryFirstChar)}" ${closeChainId}`)
      }
      for (const ac of aliasChains) {
        if (ac.chainId !== '') {
          ltAlts.push(`"${escapeGbnfString(closeBranch)}${escapeGbnfString(ac.firstChar)}" ${ac.chainId}`)
        }
      }
      // Divergence after '</' with a char that's neither primary nor alias first char
      const slashExcludeChars = [
        ...(closeChainId !== '' ? [escapeGbnfCharClass(closeChars[2])] : []),
        ...aliasChains.map(ac => escapeGbnfCharClass(ac.firstChar)),
      ].join('')
      if (slashExcludeChars.length > 0) {
        ltAlts.push(`"${escapeGbnfString(closeBranch)}" [^${slashExcludeChars}]`)
      }
    }
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
  // e.g., for close tag without leading '<', also stop at the close tag without '<'
  // This adds '/' as a top-level branch in the BUC (not just inside the lt sub-rule)
  if (options?.afterLt) {
    // Build chain for close tag without '</' — just 'magnitude parameter name="command">'
    const slashCloseSeq = options?.excludeClosePrefix
      ? '/' + options.excludeClosePrefix
      : '/' + closeTagName + '>'
    const slashCloseChars = slashCloseSeq.split('')
    // Chain starts after '/' — check 'magnitude parameter name="command">'
    const slashCloseChainId = buildChain(`${prefix}-sc`, slashCloseChars.slice(1))

    // Also stop at open prefix without '<' (e.g., 'magnitude filter>' when excludeOpenPrefix)
    // Build chain for open tag without '<' — e.g., 'magnitude filter>'
    const openWithoutLtSeq = options?.excludeOpenPrefix ? openTagName : openTagName + '>'
    const openWithoutLtChars = openWithoutLtSeq.split('')
    const openWithoutLtBranch = openWithoutLtChars[0] // 'm' for 'magnitude filter>'
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
