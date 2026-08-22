import { useCallback, useMemo } from 'react'
import { Option } from 'effect'
import { Files, useAgentClient, selectedFilePathAtom } from '@magnitudedev/client-common'
import { Atom, Result, useAtomValue, useAtomSet } from '@effect-atom/atom-react'
import type { ReadFileResult, ResolvePathResult } from '@magnitudedev/sdk'
import { selectedFileSectionAtom } from '../state/cli-atoms'
import type { TurnState } from '../utils/file-panel-utils'
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
      streamingTarget?: 'old' | 'new'
      baseContent: string | null
    }

export interface UseFilePanelParams {
  cwd: string | null
  toolState: TurnState | null
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

/** Null while nothing is selected or the files service is not ready. */
const idleAtom = Atom.make(() => null)

/**
 * The selected file's resolution and content are contract queries kept fresh
 * by the `Files` service while observed; this hook only selects and derives.
 */
export function useFilePanel({
  cwd,
  toolState,
}: UseFilePanelParams): UseFilePanelResult {
  const client = useAgentClient()
  const files = useMemo(() => client.runtime.atom(Files), [client])

  // Selected file is atom state: the path is the shared selectedFilePathAtom
  // (any feature can open a file), the section anchor is CLI-only.
  const selectedFilePath = useAtomValue(selectedFilePathAtom)
  const setSelectedFilePath = useAtomSet(selectedFilePathAtom)
  const selectedFileSection = useAtomValue(selectedFileSectionAtom)
  const setSelectedFileSection = useAtomSet(selectedFileSectionAtom)
  const selectedFile = useMemo<SelectedFileRef | null>(
    () => (selectedFilePath ? { path: selectedFilePath, section: selectedFileSection } : null),
    [selectedFilePath, selectedFileSection],
  )

  const resolveAtom = useMemo(
    () => selectedFilePath && cwd
      ? Atom.make((get): Result.Result<ResolvePathResult, unknown> | null =>
          Option.match(Result.value(get(files)), {
            onNone: () => null,
            onSome: (service) => get(service.resolve({ cwd, path: selectedFilePath })).result,
          }))
      : idleAtom,
    [files, cwd, selectedFilePath],
  )
  const resolveResult = useAtomValue(resolveAtom)
  const selectedFileResolvedPath = resolveResult !== null
    && Result.isSuccess(resolveResult)
    && resolveResult.value.exists
      ? resolveResult.value.resolved
      : null

  const readAtom = useMemo(
    () => selectedFilePath && cwd && selectedFileResolvedPath !== null
      ? Atom.make((get): Result.Result<ReadFileResult, unknown> | null =>
          Option.match(Result.value(get(files)), {
            onNone: () => null,
            onSome: (service) => get(service.read({ cwd, path: selectedFilePath, format: 'text' })).result,
          }))
      : idleAtom,
    [files, cwd, selectedFilePath, selectedFileResolvedPath],
  )
  const readResult = useAtomValue(readAtom)
  const selectedFileContent = readResult !== null && Result.isSuccess(readResult)
    ? readResult.value.content
    : null

  const activeStream = useMemo(() => {
    if (!selectedFile || !toolState) return null

    const toolHandles = toolState?.handles?.handles
      ? Object.fromEntries(toolState.handles.handles)
      : undefined
    const byRaw = findActiveFileStream(toolHandles, selectedFile.path)
    if (byRaw) return byRaw
    if (selectedFileResolvedPath && selectedFileResolvedPath !== selectedFile.path) {
      return findActiveFileStream(toolHandles, selectedFileResolvedPath)
    }
    return null
  }, [selectedFile, selectedFileResolvedPath, toolState])

  // The read query is live; the latest content is the disk content.
  const readSelectedFile = useCallback(() => selectedFileContent, [selectedFileContent])

  const frozenBaseContent = useFrozenBaseContent(
    activeStream ? { toolCallId: activeStream.toolCallId } : null,
    selectedFileContent,
    selectedFileResolvedPath,
    readSelectedFile,
  )

  const selectedFileStreaming = useMemo<FilePanelStream | null>(() => {
    if (!activeStream) return null
    const { state } = activeStream
    const status: FileOperationStatus = state.phase === 'streaming' ? 'receiving' : 'applying'

    const frozenBase = frozenBaseContent ?? selectedFileContent

    if ('body' in state) {
      return {
        mode: 'write' as const,
        status,
        body: state.body,
        baseContent: frozenBase,
      }
    }

    if ('oldText' in state) {
      return {
        mode: 'edit' as const,
        status,
        oldText: state.oldText,
        newText: state.newText,
        replaceAll: state.replaceAll,
        baseContent: frozenBase,
        ...(state.streamingTarget ? { streamingTarget: state.streamingTarget } : {}),
      }
    }

    return null
  }, [activeStream, frozenBaseContent, selectedFileContent])

  const openFile = useCallback((path: string, section?: string) => {
    const isSame = selectedFilePath === path && selectedFileSection === section
    setSelectedFilePath(isSame ? null : path)
    setSelectedFileSection(isSame ? undefined : section)
  }, [selectedFilePath, selectedFileSection, setSelectedFilePath, setSelectedFileSection])

  const closeFilePanel = useCallback(() => {
    setSelectedFilePath(null)
    setSelectedFileSection(undefined)
  }, [setSelectedFilePath, setSelectedFileSection])

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
