/**
 * Anthropic-style function_calls XML replace format.
 * Mimics the exact XML tool invocation format Claude is trained on.
 * The model emits str_replace_editor tool calls with old_str/new_str parameters.
 */
import type { EditFormat } from './types'

interface ReplaceEdit {
  oldStr: string
  newStr: string
}

const TOOL_NAME = 'str_replace_editor'
const PARAM_OLD = 'old_str'
const PARAM_NEW = 'new_str'

function parseAnthropicToolCalls(response: string): ReplaceEdit[] {
  const edits: ReplaceEdit[] = []

  // Match <invoke name="str_replace_editor"> blocks
  const invokeRegex = new RegExp(
    '<invoke\\s+name="' + TOOL_NAME + '"[^>]*>([\\s\\S]*?)</invoke>',
    'g'
  )
  let invokeMatch: RegExpExecArray | null
  while ((invokeMatch = invokeRegex.exec(response)) !== null) {
    const body = invokeMatch[1]

    // Extract parameter values
    const oldStr = extractParam(body, PARAM_OLD)
    const newStr = extractParam(body, PARAM_NEW)

    if (oldStr !== null && newStr !== null) {
      edits.push({ oldStr, newStr })
    }
  }

  return edits
}

function extractParam(body: string, name: string): string | null {
  const regex = new RegExp(
    '<parameter\\s+name="' + name + '">((?:(?!</parameter>)[\\s\\S])*)</parameter>'
  )
  const match = regex.exec(body)
  return match ? match[1] : null
}

export const anthropicReplaceFormat: EditFormat = {
  id: 'anthropic-replace',

  formatFile(_filename: string, content: string): string {
    return content
  },

  systemInstructions(): string {
    const lines = [
      '## Edit format: XML tool calls',
      '',
      'You have access to a str_replace_editor tool. To make edits, emit tool calls in this exact XML format:',
      '',
      '<function_calls>',
      '<invoke name="str_replace_editor">',
      '<parameter name="command">str_replace</parameter>',
      '<parameter name="old_str">exact text to find</parameter>',
      '<parameter name="new_str">replacement text</parameter>',
      '</invoke>',
      '</function_calls>',
      '',
      'Rules:',
      '- old_str must exactly match a unique substring of the file',
      '- Include enough context in old_str to uniquely identify the location',
      '- Use multiple <invoke> blocks within a single <function_calls> for multiple edits',
      '',
      'Return ONLY the XML tool calls. No explanation or commentary.',
    ]
    return lines.join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const edits = parseAnthropicToolCalls(response)
    if (edits.length === 0) {
      throw new Error('No str_replace_editor tool calls found in response')
    }

    let result = originalContent

    for (const edit of edits) {
      const idx = result.indexOf(edit.oldStr)
      if (idx === -1) {
        throw new Error(`old_str not found in file: ${JSON.stringify(edit.oldStr.slice(0, 80))}`)
      }
      result = result.slice(0, idx) + edit.newStr + result.slice(idx + edit.oldStr.length)
    }

    return result
  },
}
