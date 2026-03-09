/**
 * Whole-file replacement — baseline format.
 * LLM sees raw file, returns complete corrected file.
 */
import type { EditFormat } from './types'

/**
 * Build an anchor string from the first few lines of the original file.
 * Takes consecutive lines until at least 50 chars are accumulated,
 * forming a distinctive multi-line sequence unlikely to appear in explanations.
 */
function buildAnchor(originalContent: string): string | null {
  const lines = originalContent.split('\n')
  let anchor = ''
  for (const line of lines) {
    anchor += (anchor ? '\n' : '') + line
    if (anchor.length >= 50) return anchor
  }
  return anchor.length > 0 ? anchor : null
}

/**
 * Extract the file content from a response that may contain:
 * - Leading explanation text
 * - Markdown code fences wrapping the file
 * - Or just raw file content
 *
 * Uses a multi-line anchor from the original file to find where the code starts.
 */
const FENCE = '\`\`\`'
const FENCE_OPEN_RE = new RegExp(FENCE + '\\w*\\n([\\s\\S]*?)' + FENCE, 'g')

function extractFileContent(raw: string, originalContent: string): string {
  const content = raw.trim()
  const anchor = buildAnchor(originalContent)

  // 1. If response already starts with the anchor, it's clean
  if (anchor && content.startsWith(anchor)) {
    return content
  }

  // 2. Find the anchor in the response — this handles both:
  //    - Bare preamble text followed by code
  //    - Code inside a markdown fence
  //    The multi-line anchor is distinctive enough to avoid false matches
  //    in short inline code snippets within explanation text.
  if (anchor) {
    const idx = content.indexOf(anchor)
    if (idx > 0) {
      // Check if it's inside a markdown fence
      const beforeAnchor = content.slice(0, idx)
      const lastFenceOpen = beforeAnchor.lastIndexOf(FENCE)
      if (lastFenceOpen >= 0) {
        // Anchor is inside a fenced block — find the closing fence after the anchor
        const afterAnchor = content.indexOf(FENCE, idx)
        if (afterAnchor > idx) {
          return content.slice(idx, afterAnchor)
        }
      }
      // Not in a fence — just take from anchor to end
      return content.slice(idx)
    }
  }

  // 3. Fallback: try markdown fence extraction (pick the longest block)
  const fenceMatches = [...content.matchAll(FENCE_OPEN_RE)]
  if (fenceMatches.length > 0) {
    const best = fenceMatches.reduce((a, b) => a[1].length > b[1].length ? a : b)
    return best[1]
  }

  // 4. Fallback — return as-is
  return content
}

export const wholeFormat: EditFormat = {
  id: 'whole',

  formatFile(_filename: string, content: string): string {
    return content
  },

  systemInstructions(): string {
    return [
      '## Edit format: whole-file replacement',
      '',
      'Return the COMPLETE corrected file contents.',
      'Do NOT wrap in markdown code fences.',
      'Do NOT include any explanation or commentary — only the file content.',
    ].join('\n')
  },

  applyResponse(response: string, originalContent: string): string {
    return extractFileContent(response, originalContent)
  },
}
