/**
 * XML-edit format — line-number prefixed XML operations (Magnitude's format).
 * LLM sees file with LINENUM|content prefixes, returns <replace>, <insert>, <remove> XML tags.
 *
 * Reuses parseEditOps + applyOps from packages/agent/src/util/line-edit.ts.
 */
import { parseEditOps, applyOps, formatNumberedLines } from '../../../../../packages/agent/src/util/line-edit'
import type { EditFormat } from './types'

export const xmlEditFormat: EditFormat = {
  id: 'xml-edit',

  formatFile(_filename: string, content: string): string {
    return formatNumberedLines(content)
  },

  systemInstructions(): string {
    return [
      '## Edit format: XML line operations',
      '',
      'The file is shown with `LINENUM|content` prefixes (1-indexed).',
      '',
      'Return XML edit operations:',
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
