/**
 * Aider/Cline-style SEARCH/REPLACE blocks.
 * LLM sees raw file, returns <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks.
 */
import type { EditFormat } from './types'

interface ReplaceEdit {
  oldStr: string
  newStr: string
}

function parseSearchReplaceBlocks(response: string): ReplaceEdit[] {
  const edits: ReplaceEdit[] = []
  const regex = /<<<<<<< SEARCH\n([\s\S]*?)\n=======\n([\s\S]*?)\n>>>>>>> REPLACE/g
  let match: RegExpExecArray | null
  while ((match = regex.exec(response)) !== null) {
    edits.push({ oldStr: match[1], newStr: match[2] })
  }
  return edits
}

export const aiderReplaceFormat: EditFormat = {
  id: 'aider-replace',

  formatFile(_filename: string, content: string): string {
    return content
  },

  systemInstructions(): string {
    return [
      '## Edit format: SEARCH/REPLACE blocks',
      '',
      'Return one or more SEARCH/REPLACE blocks to edit the file.',
      '',
      'Each block has this format:',
      '```',
      '<<<<<<< SEARCH',
      '[exact existing code to find]',
      '=======',
      '[replacement code]',
      '>>>>>>> REPLACE',
      '```',
      '',
      'Rules:',
      '- The SEARCH section must EXACTLY MATCH the existing file content, character for character',
      '- Include just enough context in the SEARCH section to uniquely identify the location',
      '- Only the first match of each SEARCH block will be replaced',
      '- Use multiple blocks for multiple changes',
      '',
      'Return ONLY the SEARCH/REPLACE blocks. No explanation or commentary.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    const edits = parseSearchReplaceBlocks(response)
    if (edits.length === 0) {
      throw new Error('No SEARCH/REPLACE blocks found in response')
    }

    let result = originalContent

    for (const edit of edits) {
      const idx = result.indexOf(edit.oldStr)
      if (idx === -1) {
        throw new Error(`SEARCH block not found in file: ${JSON.stringify(edit.oldStr.slice(0, 80))}`)
      }
      result = result.slice(0, idx) + edit.newStr + result.slice(idx + edit.oldStr.length)
    }

    return result
  },
}
