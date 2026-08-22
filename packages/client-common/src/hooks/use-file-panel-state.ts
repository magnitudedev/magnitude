/**
 * File panel state hook — shared container logic for the file viewer panel.
 *
 * Reads the selected file through the `Files` service: `resolve` and `read`
 * queries that stay fresh while observed (the service drains `WatchFile`).
 * All state is declarative — no useState, no useEffect, no refs. All nullable
 * values use Option.
 *
 * Design: $M/designs/file-panel-state.md
 */
import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result, Atom } from "@effect-atom/atom-react"
import { Option } from "effect"
import type { ReadFileResult, ResolvePathResult } from "@magnitudedev/sdk"
import { Files } from "../files/service"
import { useAgentClient } from "../state/agent-client-context"
import {
  selectedCwdAtom,
  selectedFilePathAtom,
} from "../state/session-atoms"
import type { FilePanelStream, FileOperationStatus } from "../utils/file-panel-utils"

export type { FilePanelStream, FileOperationStatus }

export interface UseFilePanelStateResult {
  /** Selected file path (None = panel closed) */
  readonly filePath: Option.Option<string>
  /** Resolved absolute path (None if not resolved or doesn't exist) */
  readonly resolvedPath: Option.Option<string>
  /** File content (None if not loaded) */
  readonly content: Option.Option<string>
  /** Whether the file is being loaded (first load only) */
  readonly loading: boolean
  /** Error message if ResolvePath or ReadFile failed. None otherwise. */
  readonly error: Option.Option<string>
  /**
   * Streaming preview state. None when no active tool stream targets this file.
   * Currently always None — tool handles are not yet exposed to clients.
   */
  readonly streaming: Option.Option<FilePanelStream>
  /** Close the file panel */
  readonly close: () => void
}

const IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "webp", "svg"]

/** Idle atom — null when no file is selected or the service is not ready. */
const idleAtom = Atom.make(() => null)

export function useFilePanelState(): UseFilePanelStateResult {
  const filePathRaw = useAtomValue(selectedFilePathAtom)
  const setFilePath = useAtomSet(selectedFilePathAtom)
  const client = useAgentClient()
  const selectedCwdRaw = useAtomValue(selectedCwdAtom)
  const files = useMemo(() => client.runtime.atom(Files), [client])

  const filePath = Option.fromNullable(filePathRaw)
  const selectedCwd = Option.fromNullable(selectedCwdRaw)

  // Determine format based on file extension — images need base64
  const isImageFile = Option.match(filePath, {
    onNone: () => false,
    onSome: (path) => IMAGE_EXTENSIONS.includes(path.split(".").pop() ?? ""),
  })

  // ResolvePath — live while observed, so a non-existent file recovers when created.
  const resolvePathAtom = useMemo(
    () => Option.match(Option.all([filePath, selectedCwd]), {
      onNone: () => idleAtom,
      onSome: ([path, cwd]) => Atom.make((get): Result.Result<ResolvePathResult, unknown> | null =>
        Option.match(Result.value(get(files)), {
          onNone: () => null,
          onSome: (service) => get(service.resolve({ cwd, path })).result,
        })),
    }),
    [files, selectedCwd, filePath],
  )
  const resolveResult = useAtomValue(resolvePathAtom)

  const fileExists = resolveResult !== null && Result.isSuccess(resolveResult)
    ? resolveResult.value.exists
    : false

  const resolvedPath = Option.isSome(filePath)
    && resolveResult !== null
    && Result.isSuccess(resolveResult)
    && fileExists
      ? Option.some(resolveResult.value.resolved)
      : Option.none()

  // ReadFile — gated on resolvedPath to skip the read when the file doesn't exist.
  const readFileAtom = useMemo(
    () => Option.match(Option.all([filePath, selectedCwd, resolvedPath]), {
      onNone: () => idleAtom,
      onSome: ([path, cwd]) => Atom.make((get): Result.Result<ReadFileResult, unknown> | null =>
        Option.match(Result.value(get(files)), {
          onNone: () => null,
          onSome: (service) => get(service.read({
            cwd,
            path,
            format: isImageFile ? "base64" : "text",
          })).result,
        })),
    }),
    [files, selectedCwd, filePath, resolvedPath, isImageFile],
  )
  const readResult = useAtomValue(readFileAtom)

  // All states are pure derivations of Result
  const loading = Option.isSome(filePath)
    && Option.isSome(selectedCwd)
    && readResult !== null
    && Result.isInitial(readResult)

  const error: Option.Option<string> = (() => {
    if (Option.isNone(filePath) || Option.isNone(selectedCwd)) return Option.none()
    if (resolveResult !== null && Result.isFailure(resolveResult))
      return Option.some("Failed to resolve file path.")
    if (resolveResult !== null && Result.isSuccess(resolveResult) && !fileExists)
      return Option.some("File does not exist.")
    if (readResult !== null && Result.isFailure(readResult))
      return Option.some("Failed to read file. The file may not exist or is not accessible.")
    return Option.none()
  })()

  const content = Option.isSome(filePath)
    && readResult !== null
    && Result.isSuccess(readResult)
      ? Option.some(readResult.value.content)
      : Option.none()

  return {
    filePath,
    resolvedPath,
    content,
    loading,
    error,
    streaming: Option.none(),
    close: () => setFilePath(null),
  }
}
