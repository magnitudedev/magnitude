import type { Format, Scenario, Scores } from './types'

// --- Syntax validity ---

function hasXmlSyntax(output: string): boolean {
  const hasReasoning = /<reasoning>[\s\S]*?<\/reasoning>/.test(output)
  const hasActions = /<actions[\s\S]*?<\/actions>/.test(output) || /<actions\s*\/>/.test(output)
  const hasTurnControl = /<(next|yield)\s*\/>/.test(output)
  return hasReasoning && hasActions && hasTurnControl
}

function hasXmlV2Syntax(output: string): boolean {
  const hasLenses = /<lenses>[\s\S]*?<\/lenses>/.test(output)
  if (!hasLenses) return false
  const hasTurnWork = /<turn:work>[\s\S]*<\/turn:work>/.test(output)
  const hasTurnAsk = /<turn:ask>[\s\S]*<\/turn:ask>/.test(output)
  const hasTurnAnswer = /<turn:answer>[\s\S]*<\/turn:answer>/.test(output)
  if (hasTurnAsk || hasTurnAnswer) return true
  if (hasTurnWork) {
    const hasDeclare = /<declare>[\s\S]*?<\/declare>/.test(output)
    const hasObserve = /<observe\s*\/>/.test(output)
    const hasConclude = /<conclude>[\s\S]*?<\/conclude>/.test(output)
    return hasDeclare && (hasObserve || hasConclude)
  }
  // Fallback: accept without turn wrapper if it has the right inner structure
  const hasDeclare = /<declare>[\s\S]*?<\/declare>/.test(output)
  const hasObserve = /<observe\s*\/>/.test(output)
  const hasConclude = /<conclude>[\s\S]*?<\/conclude>/.test(output)
  const hasAsk = /<ask>[\s\S]*?<\/ask>/.test(output)
  if (hasAsk) return true
  if (hasDeclare) return hasObserve || hasConclude
  return hasConclude
}

function isSingleTurn(format: Format, output: string, strOpen: string): boolean {
  if (format === 'xml-v2') {
    const workCount = (output.match(/<turn:work>/g) ?? []).length
    const askCount = (output.match(/<turn:ask>/g) ?? []).length
    const answerCount = (output.match(/<turn:answer>/g) ?? []).length
    return (workCount + askCount + answerCount) <= 1
  }
  if (format === 'xml-act') {
    return (output.match(/<reasoning>/g) ?? []).length <= 1
  }
  // Count think #[ occurrences
  const esc = strOpen.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return (output.match(new RegExp(`\\bthink\\s+${esc}`, 'g')) ?? []).length <= 1
}

function hasDeclareSyntax(output: string, strOpen: string, strClose: string): boolean {
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const o = esc(strOpen)
  const c = esc(strClose)
  const startsWithThink = new RegExp(`^\\s*think\\s+${o}`).test(output)
  const hasSend = new RegExp(`\\bsend\\s+${o}`).test(output)
  const hasDeclareObserve = new RegExp(`\\bdeclare\\s+${o}[\\s\\S]*?${c}\\s*do\\s*\\{[\\s\\S]*?\\}\\s*observe\\s+@`).test(output)
  const hasDeclareConc = new RegExp(`\\bdeclare\\s+${o}[\\s\\S]*?${c}\\s*do\\s*\\{[\\s\\S]*?\\}\\s*conclude\\s+${o}`).test(output)
  const opens = (output.match(new RegExp(o, 'g')) ?? []).length
  const closes = (output.match(new RegExp(c, 'g')) ?? []).length
  return startsWithThink && (hasSend || hasDeclareObserve || hasDeclareConc) && opens === closes
}

// --- Turn control ---

function usesContinuation(format: Format, output: string): boolean {
  if (format === 'xml-act') return /<next\s*\/>/.test(output)
  if (format === 'xml-v2') return /<observe\s*\/>/.test(output)
  return /\}\s*observe\s+@/.test(output)
}

// --- Progressive disclosure ---

function usesPd(format: Format, output: string): boolean {
  if (format === 'xml-act') {
    return /<inspect>[\s\S]*?<ref\b[\s\S]*?<\/inspect>/.test(output)
  }
  if (format === 'xml-v2') {
    return /\bobserve="[^"]*"/.test(output)
  }
  return /\[[^\]]*\]\s*->\s*@/.test(output)
}

// --- Hallucination detection ---

function hallucinatesResults(format: Format, output: string, strOpen: string, strClose: string): boolean {
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const o = esc(strOpen)
  const c = esc(strClose)

  // Get ALL user-facing text from the response
  let userText = ''
  if (format === 'xml-act') {
    userText = output.match(/<message>([\s\S]*?)<\/message>/)?.[1] ?? ''
  } else if (format === 'xml-v2') {
    const answerText = output.match(/<turn:answer>[\s\S]*?<\/lenses>([\s\S]*?)<\/turn:answer>/)?.[1]?.trim() ?? ''
    const concludeText = output.match(/<conclude>([\s\S]*?)<\/conclude>/)?.[1] ?? ''
    const askText = output.match(/<turn:ask>[\s\S]*?<\/lenses>([\s\S]*?)<\/turn:ask>/)?.[1]?.trim() ?? ''
    userText = answerText + ' ' + concludeText + ' ' + askText
  } else {
    const sendText = output.match(new RegExp(`\\bsend\\s+${o}([\\s\\S]*?)${c}`))?.[1] ?? ''
    const concludeText = output.match(new RegExp(`\\bconclude\\s+${o}([\\s\\S]*?)${c}`))?.[1] ?? ''
    userText = sendText + ' ' + concludeText
  }

  const claimsResults = /\b(found|contains|shows|output|passed|failed|exit code \d|error|TODO|host|localhost|the file|the config|results?:|here'?s what)/i.test(userText)

  const usesConclude = new RegExp(`\\}\\s*conclude\\s+${o}`).test(output)
  const hasReadOrSearchOrRun = /\b(read|search|run)\b/.test(output)

  if (usesConclude && hasReadOrSearchOrRun && claimsResults) {
    return true
  }

  // Standard check: claims results without continuing
  const continues = usesContinuation(format, output)
  return claimsResults && !continues
}

// --- Main evaluator ---

export function evaluateOutput(format: Format, scenario: Scenario, rawOutput: string, strOpen = '#[', strClose = ']#'): Scores {
  const syntaxValid = format === 'xml-act' ? hasXmlSyntax(rawOutput) : format === 'xml-v2' ? hasXmlV2Syntax(rawOutput) : hasDeclareSyntax(rawOutput, strOpen, strClose)
  const continues = usesContinuation(format, rawOutput)
  // Turn control is wrong only when the choice leads to incorrect behavior:
  // - Should continue but didn't AND hallucinated results = wrong
  // - Should continue but didn't, no hallucination = acceptable (just declared intent)
  // - Shouldn't continue but did = acceptable (confirming success is fine)
  let turnCorrect = true
  if (scenario.shouldContinue && !continues) {
    // Didn't continue when it should have — only wrong if it also claimed results
    turnCorrect = !hallucinatesResults(format, rawOutput, strOpen, strClose)
  }

  const escapingCorrect = scenario.checkEscaping != null
    ? rawOutput.includes(scenario.checkEscaping)
    : null

  return {
    syntax_valid: syntaxValid,
    single_turn: isSingleTurn(format, rawOutput, strOpen),
    turn_control_correct: turnCorrect,
    pd_used: scenario.shouldUsePd !== null ? usesPd(format, rawOutput) : null,
    no_hallucinated_results: scenario.checkNoHallucination !== null ? !hallucinatesResults(format, rawOutput, strOpen, strClose) : null,
    escaping_correct: escapingCorrect,
  }
}