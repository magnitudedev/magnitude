import { useRef } from 'react'

/**
 * Manages frozen base content for the file panel.
 *
 * When a stream starts, freezes the current file content so the panel
 * can compute optimistic previews against a stable base.
 *
 * No useEffect — uses a ref-based imperative pattern that executes
 * during render when the active stream changes.
 */
export function useFrozenBaseContent(
  activeStream: { toolCallId: string } | null,
  selectedFileContent: string | null,
  selectedFileResolvedPath: string | null,
  readFromDisk: () => string | null | Promise<string | null>,
): string | null {
  const frozenBaseRef = useRef<{ toolCallId: string; content: string | null } | null>(null)
  const previousStreamRef = useRef<{ toolCallId: string } | null>(null)
  const prevResolvedPathRef = useRef<string | null>(null)

  // Reset frozen base when the selected file changes
  if (prevResolvedPathRef.current !== selectedFileResolvedPath) {
    prevResolvedPathRef.current = selectedFileResolvedPath
    frozenBaseRef.current = null
    previousStreamRef.current = null
  }

  // Track stream lifecycle transitions
  const previous = previousStreamRef.current
  const current = activeStream

  if (previous !== current || (current && previous?.toolCallId !== current.toolCallId)) {
    if (!previous && current) {
      // Stream started — freeze current content
      frozenBaseRef.current = {
        toolCallId: current.toolCallId,
        content: selectedFileContent,
      }
    } else if (previous && !current) {
      // Stream ended — unfreeze and re-read from disk
      frozenBaseRef.current = null
      void readFromDisk()
    } else if (previous && current && previous.toolCallId !== current.toolCallId) {
      // Different stream — re-read disk for new base, then freeze
      const result = readFromDisk()
      Promise.resolve(result).then((nextContent: string | null) => {
        frozenBaseRef.current = {
          toolCallId: current.toolCallId,
          content: nextContent,
        }
      })
    }

    previousStreamRef.current = current ? { toolCallId: current.toolCallId } : null
  }

  // Return frozen content only if it matches the current stream
  if (!activeStream || !frozenBaseRef.current) return null
  if (frozenBaseRef.current.toolCallId !== activeStream.toolCallId) return null
  return frozenBaseRef.current.content
}
