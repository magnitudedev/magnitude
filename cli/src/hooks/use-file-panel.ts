import { useCallback, useEffect, useMemo, useState } from 'react'
import { resolveFileRef } from '@magnitudedev/agent'
import { logger } from '@magnitudedev/logger'
import { readFileSync, watchFile, unwatchFile } from 'node:fs'

export interface SelectedFileRef {
  path: string
  section?: string
}

export interface UseFilePanelParams {
  workspacePath: string | null
  projectRoot: string
}

export interface UseFilePanelResult {
  selectedFile: SelectedFileRef | null
  selectedFileContent: string | null
  selectedFileResolvedPath: string | null
  isOpen: boolean
  openFile: (path: string, section?: string) => void
  closeFilePanel: () => void
}

export function useFilePanel({
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

  useEffect(() => {
    void readSelectedFileFromDisk()
  }, [selectedFileResolvedPath, readSelectedFileFromDisk])

  useEffect(() => {
    if (!selectedFileResolvedPath) return

    const onFileChanged = () => {
      void readSelectedFileFromDisk()
    }

    watchFile(selectedFileResolvedPath, { interval: 500 }, onFileChanged)
    return () => {
      unwatchFile(selectedFileResolvedPath, onFileChanged)
    }
  }, [selectedFileResolvedPath, readSelectedFileFromDisk])

  const openFile = useCallback((path: string, section?: string) => {
    setSelectedFile(prev => prev?.path === path && prev?.section === section ? null : { path, section })
  }, [])

  const closeFilePanel = useCallback(() => {
    setSelectedFile(null)
  }, [])

  const isOpen = selectedFile != null
  return {
    selectedFile,
    selectedFileContent,
    selectedFileResolvedPath,
    isOpen,
    openFile,
    closeFilePanel,
  }
}
