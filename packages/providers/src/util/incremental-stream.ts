/**
 * Converts a cumulative stream (where each chunk contains all previous content)
 * into an incremental stream (where each chunk contains only new content)
 *
 * Example:
 * Input:  "a" -> "ab" -> "abc" -> "abcd"
 * Output: "a" -> "b" -> "c" -> "d"
 */
export async function* toIncrementalStream(
  cumulativeStream: AsyncIterable<string>
): AsyncGenerator<string> {
  let previousContent = ''

  for await (const cumulativeChunk of cumulativeStream) {
    if (cumulativeChunk.length > previousContent.length) {
      const newContent = cumulativeChunk.substring(previousContent.length)
      yield newContent
    }
    previousContent = cumulativeChunk
  }
}
