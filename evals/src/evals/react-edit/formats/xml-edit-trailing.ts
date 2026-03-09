/**
 * XML-edit-trailing format — trailing line-number suffixed XML operations.
 * LLM sees file with content|LINENUM suffixes (line number AFTER code),
 * returns the same <replace>, <insert>, <remove> XML tags as xml-edit.
 *
 * Hypothesis: trailing line numbers interfere less with token prediction
 * since the code tokens appear first in context.
 */
import { parseEditOps, applyOps } from '../../../../../packages/agent/src/util/line-edit'
import type { EditFormat } from './types'

function formatTrailingNumberedLines(content: string): string {
  const lines = content.split('\n')
  return lines
    .map((line, i) => `${line} |${i + 1}`)
    .join('\n')
}

export const xmlEditTrailingFormat: EditFormat = {
  id: 'xml-edit-trailing',

  formatFile(_filename: string, content: string): string {
    return formatTrailingNumberedLines(content)
  },

  systemInstructions(): string {
    return [
      '## Edit format: XML line operations (trailing line numbers)',
      '',
      'The file is shown with trailing line numbers: each line ends with ` |LINENUM` (1-indexed).',
      'Read the number at the END of each line to determine its line number.',
      '',
      'Return XML edit operations using those line numbers:',
      '',
      '**Replace lines:**',
      '```xml',
      '<replace from=N to=M>',
      'new content here',
      '</replace>',
      '```',
      '',
      '**Insert after a line:**',
      '```xml',
      '<insert after=N>',
      'new content here',
      '</insert>',
      '```',
      '',
      '**Remove lines:**',
      '```xml',
      '<remove from=N to=M />',
      '```',
      '',
      'Line numbers refer to the original file as shown. Use `from` and `to` (inclusive, 1-based).',
      '',
      'Return ONLY the XML operations. No explanation or commentary.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const ops = parseEditOps(response)
    if (ops.length === 0) {
      throw new Error('No edit operations found in response')
    }
    const result = applyOps(originalContent, ops)
    return result.content
  },
}
