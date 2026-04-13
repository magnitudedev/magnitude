import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  resolveFileRef,
  type DisplayState,
  type ToolStateProjectionState,
} from '@magnitudedev/agent'
import { logger } from '@magnitudedev/logger'
import { readFileSync, watchFile, unwatchFile } from 'node:fs'
import { useFrozenBaseContent } from './use-frozen-base-content'
import { findActiveFileStream } from '../utils/file-panel-utils'

export interface SelectedFileRef {
  path: string
  section?: string
}

export type FileOperationStatus = 'receiving' | 'applying'

export type FilePanelStream =
  | {
      mode: 'write'
      status: FileOperationStatus
      body: string
      baseContent: string | null
    }
  | {
      mode: 'edit'
      status: FileOperationStatus
      oldText: string
      newText: string
      replaceAll: boolean
      streamingTarget: 'old' | 'new' | null
      baseContent: string | null
    }

export interface UseFilePanelParams {
  display: DisplayState | null
  toolState: ToolStateProjectionState | null
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
  toolState,
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

  const activeStream = useMemo(() => {
    if (!selectedFile || !display) return null

    const toolHandles = toolState?.toolHandles
    const byRaw = findActiveFileStream(toolHandles, selectedFile.path)
    if (byRaw) return byRaw
    if (selectedFileResolvedPath && selectedFileResolvedPath !== selectedFile.path) {
      return findActiveFileStream(toolHandles, selectedFileResolvedPath)
    }
    return null
  }, [selectedFile, selectedFileResolvedPath, display, toolState])

  const hasActiveSelectedFileStreamRef = useRef(false)
  useEffect(() => {
    hasActiveSelectedFileStreamRef.current = activeStream != null
  }, [activeStream])

  useEffect(() => {
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

  const frozenBaseContent = useFrozenBaseContent(
    activeStream ? { toolCallId: activeStream.toolCallId } : null,
    selectedFileContent,
    selectedFileResolvedPath,
    readSelectedFileFromDisk,
  )

  const selectedFileStreaming = useMemo<FilePanelStream | null>(() => {
    if (!activeStream) return null
    const { state } = activeStream
    const status: FileOperationStatus = state.phase === 'streaming' ? 'receiving' : 'applying'

    const frozenBase = frozenBaseContent ?? selectedFileContent

    if (state.toolKey === 'fileWrite') {
      return {
        mode: 'write',
        status,
        body: state.body,
        baseContent: frozenBase,
      }
    }

    return {
      mode: 'edit',
      status,
      oldText: state.oldText,
      newText: state.newText,
      replaceAll: state.replaceAll,
      streamingTarget: state.streamingTarget,
      baseContent: frozenBase,
    }
  }, [activeStream, frozenBaseContent, selectedFileContent])

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
