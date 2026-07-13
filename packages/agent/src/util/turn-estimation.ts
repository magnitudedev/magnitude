import type { CompletedTurn } from '../window/types'
import type { ToolResult } from '@magnitudedev/harness'
import { Option } from 'effect'
import { renderFeedbackText } from '../prompts/feedback-text'
import { estimateText, estimateImageTokens, DEFAULT_IMAGE_TOKENS } from '../truncation/estimate'
import { describeShape } from '../truncation'
import { TRUNCATION_TOKEN_LIMIT } from '../constants'

function estimateResultTokens(result: ToolResult): number {
  switch (result._tag) {
    case "Success": {
      if (result.output === undefined) return 10
      try {
        const serialized = JSON.stringify(result.output)
        const estimated = estimateText(serialized)
        // Account for truncation — large outputs get replaced with a short summary
        if (estimated > TRUNCATION_TOKEN_LIMIT) {
          return estimateText(describeShape(result.output)) + 50
        }
        return estimated
      } catch {
        return 50
      }
    }
    case "Error":
      return estimateText(result.error.message) + 30
    case "Denied":
      return estimateText(typeof result.denial === 'string' ? result.denial : JSON.stringify(result.denial)) + 30
    case "Interrupted":
      return 10
    case "InputRejected":
      return estimateText(JSON.stringify(result.partialInput)) + 80
  }
}

export function estimateCompletedTurn(turn: CompletedTurn): number {
  let tokens = 0

  // Assistant reasoning + text
  tokens += estimateText(Option.getOrElse(turn.assistant.reasoning, () => ''))
  tokens += estimateText(Option.getOrElse(turn.assistant.text, () => ''))

  // Tool calls: input JSON + framing overhead
  const toolCalls = Option.getOrElse(turn.assistant.toolCalls, () => [])
  if (toolCalls.length > 0) {
    for (const tc of toolCalls) {
      tokens += estimateText(tc.name)
      tokens += estimateText(JSON.stringify(tc.input))
      tokens += 20 // JSON framing overhead per tool call
    }
  }

  // Tool results (semantic outcomes)
  for (const entry of turn.toolResults) {
    tokens += estimateResultTokens(entry.result)
  }

  // Feedback
  tokens += estimateText(renderFeedbackText(turn.feedback))

  return tokens
}
