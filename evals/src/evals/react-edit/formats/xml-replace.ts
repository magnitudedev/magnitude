/**
 * XML replace format — simple <edit> with <old>/<new> content blocks.
 * Essentially the same as anthropic-replace but with minimal XML structure
 * instead of function_calls/invoke/parameter nesting.
 */
import type { EditFormat } from './types'

interface ReplaceEdit {
  oldStr: string
  newStr: string
}

function parseXmlReplaceBlocks(response: string): ReplaceEdit[] {
  const edits: ReplaceEdit[] = []

  const editRegex = /<edit[^>]*>([\s\S]*?)<\/edit>/g
  let editMatch: RegExpExecArray | null
  while ((editMatch = editRegex.exec(response)) !== null) {
    const body = editMatch[1]

    const oldMatch = /<old>([\s\S]*?)<\/old>/.exec(body)
    const newMatch = /<new>([\s\S]*?)<\/new>/.exec(body)

    if (oldMatch && newMatch) {
      // Strip leading/trailing newline from tag content
      const oldStr = oldMatch[1].replace(/^\n/, '').replace(/\n$/, '')
      const newStr = newMatch[1].replace(/^\n/, '').replace(/\n$/, '')
      edits.push({ oldStr, newStr })
    }
  }

  return edits
}

export const xmlReplaceFormat: EditFormat = {
  id: 'xml-replace',

  formatFile(_filename: string, content: string): string {
    return content
  },

  systemInstructions(): string {
    return [
      '## Edit format: XML replace',
      '',
      'To edit the file, return one or more <edit> blocks with <old> and <new> sections:',
      '',
      '<edit>',
      '<old>',
      'exact existing code to find',
      '</old>',
      '<new>',
      'replacement code',
      '</new>',
      '</edit>',
      '',
      'Rules:',
      '- The <old> content must EXACTLY MATCH the file — character for character, including indentation and whitespace',
      '- The <old> content must appear only once in the file. Include enough surrounding lines to make it unique.',
      '- Use multiple <edit> blocks for multiple changes',
      '',
      'Return ONLY the <edit> blocks. No explanation or commentary.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const edits = parseXmlReplaceBlocks(response)
    if (edits.length === 0) {
      throw new Error('No <edit> blocks found in response')
    }

    let result = originalContent

    for (const edit of edits) {
      const idx = result.indexOf(edit.oldStr)
      if (idx === -1) {
        throw new Error(`<old> content not found in file: ${JSON.stringify(edit.oldStr.slice(0, 80))}`)
      }
      result = result.slice(0, idx) + edit.newStr + result.slice(idx + edit.oldStr.length)
    }

    return result
  },
}
