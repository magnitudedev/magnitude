/**
 * Stream Guard
 *
 * Truncates an LLM output stream after a structural closing tag is encountered.
 * Also injects the provided closing tag if the stream ends with an opening tag
 * but no matching close (e.g. truncation).
 */

import { Stream, Effect } from 'effect'

function hasLeadingBoundary(text: string, index: number): boolean {
  return index === 0 || text[index - 1] === '\n'
}

function hasTrailingBoundary(text: string, index: number, tagLength: number): boolean {
  const end = index + tagLength
  return end === text.length || text[end] === '\n'
}

function isStructuralMatch(text: string, index: number, tagLength: number): boolean {
  // Match parser's "either/or" style boundary tolerance:
  // structural tags are accepted if they have either a leading boundary
  // (start/newline) OR a trailing boundary (newline/end).
  return hasLeadingBoundary(text, index) || hasTrailingBoundary(text, index, tagLength)
}

function findStructuralTag(text: string, tag: string): number {
  let from = 0
  while (from <= text.length - tag.length) {
    const idx = text.indexOf(tag, from)
    if (idx === -1) return -1
    if (isStructuralMatch(text, idx, tag.length)) return idx
    from = idx + 1
  }
  return -1
}

function hasStructuralTag(text: string, tag: string): boolean {
  return findStructuralTag(text, tag) !== -1
}

export async function* guardStream(
  source: AsyncIterator<string>,
  closingTag: string,
  openingTag: string,
): AsyncGenerator<string> {
  let accumulated = ''
  let sourceClosed = false

  try {
    while (true) {
      const next = await source.next()

      if (next.done) {
        if (hasStructuralTag(accumulated, openingTag) && !hasStructuralTag(accumulated, closingTag)) {
          yield closingTag
        }
        return
      }

      const chunk = next.value
      accumulated += chunk

      const closingIdx = findStructuralTag(accumulated, closingTag)
      if (closingIdx !== -1) {
        const alreadyYielded = accumulated.length - chunk.length
        const endIdx = closingIdx + closingTag.length
        const toYield = endIdx - alreadyYielded

        if (toYield > 0) {
          yield chunk.slice(0, toYield)
        }

        await source.return?.(undefined as never)
        sourceClosed = true
        return
      }

      yield chunk
    }
  } finally {
    if (!sourceClosed) {
      await source.return?.(undefined as never)
    }
  }
}

export function guardEffectStream<E, R>(
  source: Stream.Stream<string, E, R>,
  closingTag: string,
  openingTag: string,
): Stream.Stream<string, E, R> {
  let accumulated = ''
  let stopped = false

  const guarded = source.pipe(
    Stream.takeWhile(() => !stopped),
    Stream.map((chunk) => {
      accumulated += chunk
      const idx = findStructuralTag(accumulated, closingTag)
      if (idx !== -1) {
        stopped = true
        const endIdx = idx + closingTag.length
        const alreadyYielded = accumulated.length - chunk.length
        const toYield = chunk.substring(0, endIdx - alreadyYielded)
        return toYield
      }
      return chunk
    }),
    Stream.filter((chunk) => chunk.length > 0),
  )

  return Stream.concat(
    guarded,
    Stream.fromEffect(
      Effect.sync(() => {
        if (hasStructuralTag(accumulated, openingTag) && !hasStructuralTag(accumulated, closingTag)) {
          return closingTag
        }
        return ''
      }),
    ).pipe(Stream.filter((s) => s.length > 0)),
  )
}