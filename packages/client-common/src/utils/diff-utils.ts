export interface MatchRange {
  start: number
  end: number
}

export function findUniqueMatchRange(content: string | null | undefined, needle: string | undefined): MatchRange | null {
  if (!content || !needle) return null
  const first = content.indexOf(needle)
  if (first === -1) return null
  const second = content.indexOf(needle, first + 1)
  if (second !== -1) return null
  return { start: first, end: first + needle.length }
}
