import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getLatestInProgressFileStream, resolveFileRef, type DisplayState } from '@magnitudedev/agent'
import { logger } from '@magnitudedev/logger'
import { readFileSync, watchFile, unwatchFile } from 'node:fs'

export interface SelectedFileRef {
  path: string
  section?: string
}

export type FileOperationStatus = 'receiving' | 'applying'

export type FilePanelStream =
  | {
      mode: 'write'
      status: FileOperationStatus
      contentSoFar: string
      baseContent: string | null
    }
  | {
      mode: 'replace'
      status: FileOperationStatus
      oldStringSoFar: string
      newStringSoFar: string
      replaceAll: boolean
      baseContent: string | null
    }

export interface UseFilePanelParams {
  display: DisplayState | null
  workspacePath: string | null
  projectRoot: string
}

export interface UseFilePanelResult {
  selectedFile: SelectedFileRef | null
  selectedFileContent: string | null
  selectedFileStreaming: FilePanelStream | null
  selectedFileResolvedPath: string | null
  isOpen: boolean
  canRenderPanel: boolean
  openFile: (path: string, section?: string) => void
  closeFilePanel: () => void
}

export function useFilePanel({
  display,
  workspacePath,
  projectRoot,
}: UseFilePanelParams): UseFilePanelResult {
  const [selectedFile, setSelectedFile] = useState<SelectedFileRef | null>(null)
  const [selectedFileContent, setSelectedFileContent] = useState<string | null>(null)

  const selectedFileResolvedPath = useMemo(() => {
    if (!selectedFile) return null
    if (!workspacePath) {
      logger.warn({ path: selectedFile.path, cwd: projectRoot }, 'workspacePath not set while resolving selected file; session likely not initialized')
      return null
    }

    const resolved = resolveFileRef(selectedFile.path, projectRoot, workspacePath)
    if (!resolved?.resolvedPath) {
      logger.warn({ path: selectedFile.path, cwd: projectRoot, workspacePath }, 'resolveFileRef returned null for selected file')
      return null
    }
    return resolved.resolvedPath
  }, [selectedFile, workspacePath, projectRoot])

  const readSelectedFileFromDisk = useCallback(() => {
    if (!selectedFile || !selectedFileResolvedPath) {
      setSelectedFileContent(null)
      return null
    }

    try {
      const nextContent = readFileSync(selectedFileResolvedPath, 'utf-8')
      setSelectedFileContent(nextContent)
      return nextContent
    } catch (err) {
      logger.error({ path: selectedFile.path, cwd: projectRoot, workspacePath, error: err instanceof Error ? err.message : String(err) }, 'Failed to read selected file')
      setSelectedFileContent(null)
      return null
    }
  }, [selectedFile, selectedFileResolvedPath, projectRoot, workspacePath])

  const selectedFileStream = useMemo(() => {
    if (!selectedFile || !display) return null
    // Try raw reference path first (matches when tool input uses same relative path)
    const byRawPath = getLatestInProgressFileStream(display, selectedFile.path)
    if (byRawPath) return byRawPath
    // Fall back to resolved absolute path (matches when tool input uses absolute path)
    if (selectedFileResolvedPath && selectedFileResolvedPath !== selectedFile.path) {
      return getLatestInProgressFileStream(display, selectedFileResolvedPath)
    }
    return null
  }, [selectedFile, selectedFileResolvedPath, display])

  const hasActiveSelectedFileStreamRef = useRef(false)
  useEffect(() => {
    hasActiveSelectedFileStreamRef.current = selectedFileStream != null
  }, [selectedFileStream])

  const frozenBaseContentRef = useRef<{ toolCallId: string; content: string | null } | null>(null)

  useEffect(() => {
    frozenBaseContentRef.current = null
    void readSelectedFileFromDisk()
  }, [selectedFileResolvedPath, readSelectedFileFromDisk])

  useEffect(() => {
    if (!selectedFileResolvedPath) return

    const onFileChanged = () => {
      if (hasActiveSelectedFileStreamRef.current) return
      void readSelectedFileFromDisk()
    }

    watchFile(selectedFileResolvedPath, { interval: 500 }, onFileChanged)
    return () => {
      unwatchFile(selectedFileResolvedPath, onFileChanged)
    }
  }, [selectedFileResolvedPath, readSelectedFileFromDisk])

  const previousSelectedFileStreamRef = useRef<{ toolCallId: string } | null>(null)
  useEffect(() => {
    const previous = previousSelectedFileStreamRef.current
    const current = selectedFileStream

    if (!previous && current) {
      frozenBaseContentRef.current = {
        toolCallId: current.toolCallId,
        content: selectedFileContent,
      }
    } else if (previous && !current) {
      frozenBaseContentRef.current = null
      void readSelectedFileFromDisk()
    } else if (previous && current && previous.toolCallId !== current.toolCallId) {
      const nextContent = readSelectedFileFromDisk()
      frozenBaseContentRef.current = {
        toolCallId: current.toolCallId,
        content: nextContent,
      }
    }

    previousSelectedFileStreamRef.current = current ? { toolCallId: current.toolCallId } : null
  }, [selectedFileStream, selectedFileContent, readSelectedFileFromDisk])

  const selectedFileStreaming = useMemo<FilePanelStream | null>(() => {
    if (!selectedFileStream) return null
    if (selectedFileStream.phase !== 'streaming' && selectedFileStream.phase !== 'executing') {
      return null
    }

    const frozenBase = frozenBaseContentRef.current?.toolCallId === selectedFileStream.toolCallId
      ? frozenBaseContentRef.current.content
      : selectedFileContent
    const status: FileOperationStatus = selectedFileStream.phase === 'streaming' ? 'receiving' : 'applying'

    if (selectedFileStream.preview.mode === 'write') {
      return {
        mode: 'write',
        status,
        contentSoFar: selectedFileStream.preview.contentSoFar,
        baseContent: frozenBase,
      }
    }

    return {
      mode: 'replace',
      status,
      oldStringSoFar: selectedFileStream.preview.oldStringSoFar,
      newStringSoFar: selectedFileStream.preview.newStringSoFar,
      replaceAll: selectedFileStream.preview.replaceAll,
      baseContent: frozenBase,
    }
  }, [selectedFileStream, selectedFileContent])

  const openFile = useCallback((path: string, section?: string) => {
    setSelectedFile(prev => prev?.path === path && prev?.section === section ? null : { path, section })
  }, [])

  const closeFilePanel = useCallback(() => {
    setSelectedFile(null)
  }, [])

  const isOpen = selectedFile != null
  const canRenderPanel = selectedFile != null && (selectedFileContent !== null || selectedFileStreaming !== null)

  return {
    selectedFile,
    selectedFileContent,
    selectedFileStreaming,
    selectedFileResolvedPath,
    isOpen,
    canRenderPanel,
    openFile,
    closeFilePanel,
  }
}
