import { useCallback, useMemo, useRef, useState } from 'react'
import { Effect, Stream, Cause, Fiber, Runtime } from 'effect'
import { useAgentClient, selectedFilePathAtom } from '@magnitudedev/client-common'
import { Result, useAtomValue, useAtomSet } from '@effect-atom/atom-react'
import { selectedFileSectionAtom } from '../state/cli-atoms'
import type { TurnState } from '../utils/file-panel-utils'
import { logger } from '@magnitudedev/logger'
import { useFrozenBaseContent } from './use-frozen-base-content'
import { findActiveFileStream } from '../utils/file-panel-utils'
import type { ReadFileResult, ResolvePathResult } from '@magnitudedev/sdk'

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

export function useFilePanel({
  cwd,
  toolState,
  projectRoot,
}: UseFilePanelParams): UseFilePanelResult {
  const atomClient = useAgentClient()
  const observationRuntimeResult = useAtomValue(atomClient.runtime)
  const resolvePathMutation = useAtomSet(atomClient.mutation('ResolvePath'), { mode: 'promise' })
  const readFileMutation = useAtomSet(atomClient.mutation('ReadFile'), { mode: 'promise' })

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
  const [selectedFileResolvedPath, setSelectedFileResolvedPath] = useState<string | null>(null)
  const [selectedFileContent, setSelectedFileContent] = useState<string | null>(null)

  // Refs to track previous values for imperative side effects (no useEffect)
  const prevFileRef = useRef<SelectedFileRef | null>(null)
  const prevResolvedPathRef = useRef<string | null>(null)
  const hasActiveStreamRef = useRef(false)
  const watchFiberRef = useRef<Fiber.RuntimeFiber<void, unknown> | null>(null)

  const readFile = useCallback(async (path: string): Promise<string | null> => {
    if (!cwd) return null
    try {
      const result = await readFileMutation({ payload: { cwd, path } }) as ReadFileResult
      return result.content
    } catch (err) {
      logger.error({ path, error: err instanceof Error ? err.message : String(err) }, 'Failed to read file')
      return null
    }
  }, [cwd, readFileMutation])

  // Resolve path when selectedFile changes — imperative during render
  if (prevFileRef.current !== selectedFile) {
    prevFileRef.current = selectedFile

    if (!selectedFile || !cwd) {
      setSelectedFileResolvedPath(null)
    } else {
      void (async () => {
        try {
          const result = await resolvePathMutation({
            payload: { cwd: cwd!, path: selectedFile.path, checkExists: true },
          }) as ResolvePathResult
          if (!result.exists) {
            logger.warn({ path: selectedFile.path, cwd: projectRoot }, 'Selected file does not exist')
            setSelectedFileResolvedPath(null)
          } else {
            setSelectedFileResolvedPath(result.resolved)
          }
        } catch (err) {
          logger.error({ path: selectedFile.path, cwd: projectRoot, error: err instanceof Error ? err.message : String(err) }, 'Failed to resolve selected file')
          setSelectedFileResolvedPath(null)
        }
      })()
    }
  }

  // Read file + set up watch when resolved path changes — imperative during render
  if (prevResolvedPathRef.current !== selectedFileResolvedPath) {
    prevResolvedPathRef.current = selectedFileResolvedPath

    // Clean up previous watch fiber
    if (watchFiberRef.current) {
      Effect.runFork(Fiber.interrupt(watchFiberRef.current))
      watchFiberRef.current = null
    }

    if (!selectedFile || !selectedFileResolvedPath || !cwd) {
      setSelectedFileContent(null)
    } else {
      const path = selectedFile.path
      const watchCwd = cwd

      // Read file
      void readFile(path).then((content) => {
        if (content !== null) setSelectedFileContent(content)
      })

      // Set up file watch via streaming RPC
      if (Result.isSuccess(observationRuntimeResult)) {
        const watchEffect = Effect.gen(function* () {
          const c = yield* atomClient
          yield* c('WatchFile', { cwd: watchCwd, path }).pipe(
            Stream.runForEach((_event) =>
              Effect.sync(() => {
                if (hasActiveStreamRef.current) return
                void readFile(path).then((content) => {
                  if (content !== null) setSelectedFileContent(content)
                })
              })
            ),
          )
        }).pipe(
          Effect.catchAllCause((cause) =>
            Cause.isInterruptedOnly(cause)
              ? Effect.void
              : Effect.sync(() => {
                  const message = Cause.pretty(cause)
                  logger.error({ path, error: message }, 'File watch error')
                })
          ),
          Effect.provide(observationRuntimeResult.value),
        )
        watchFiberRef.current = Runtime.runFork(observationRuntimeResult.value)(watchEffect)
      }
    }
  }

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

  // Track active stream via ref (no useEffect)
  hasActiveStreamRef.current = activeStream != null

  const readSelectedFile = useCallback(async () => {
    if (!selectedFile || !cwd) {
      setSelectedFileContent(null)
      return null
    }
    const content = await readFile(selectedFile.path)
    if (content !== null) setSelectedFileContent(content)
    return content
  }, [selectedFile, cwd, readFile])

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
