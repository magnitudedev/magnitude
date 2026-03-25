import { useEffect, useRef, useState } from 'react'

/**
 * Manages frozen base content for the file panel.
 * 
 * When a stream starts, freezes the current file content so the panel
 * can compute optimistic previews against a stable base.
 * 
 * Uses useState so that freeze/unfreeze transitions trigger re-renders.
 */
export function useFrozenBaseContent(
  activeStream: { toolCallId: string } | null,
  selectedFileContent: string | null,
  selectedFileResolvedPath: string | null,
  readFromDisk: () => string | null,
): string | null {
  const [frozenBase, setFrozenBase] = useState<{ toolCallId: string; content: string | null } | null>(null)
  const previousStreamRef = useRef<{ toolCallId: string } | null>(null)

  // Reset frozen base when the selected file changes
  // (must be declared before stream lifecycle effect — React runs effects in declaration order)
  useEffect(() => {
    setFrozenBase(null)
    previousStreamRef.current = null
  }, [selectedFileResolvedPath])

  // Track stream lifecycle transitions
  useEffect(() => {
    const previous = previousStreamRef.current
    const current = activeStream

    if (!previous && current) {
      // Stream started — freeze current content
      setFrozenBase({
        toolCallId: current.toolCallId,
        content: selectedFileContent,
      })
    } else if (previous && !current) {
      // Stream ended — unfreeze and re-read from disk
      setFrozenBase(null)
      readFromDisk()
    } else if (previous && current && previous.toolCallId !== current.toolCallId) {
      // Different stream — re-read disk for new base, then freeze
      const nextContent = readFromDisk()
      setFrozenBase({
        toolCallId: current.toolCallId,
        content: nextContent,
      })
    }

    previousStreamRef.current = current ? { toolCallId: current.toolCallId } : null
  }, [activeStream, selectedFileContent, readFromDisk])

  // Return frozen content only if it matches the current stream
  if (!activeStream || !frozenBase) return null
  if (frozenBase.toolCallId !== activeStream.toolCallId) return null
  return frozenBase.content
}
