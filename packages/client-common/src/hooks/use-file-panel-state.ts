/**
 * File panel state hook — shared container logic for the file viewer panel.
 *
 * Uses ResolvePath query, ReadFile query (with reactivityKeys for file watching),
 * and subscribeFileWatch for live edits. The freeze mechanism (snapshot when
 * a stream starts) is a pure derivation; the re-read when the stream ends is
 * a subscription-triggered query refresh.
 */
import { useMemo } from "react"
import { useAtomValue, useAtomSet, Result, Atom } from "@effect-atom/atom-react"
import { useAgentClient } from "../state/agent-client-context"
import { usePlatform } from "../platform/platform-context"
import { useDisplayState } from "../state/display-state-store"
import { getFork } from "../sync/get-fork"
import {
  selectedCwdAtom,
  selectedFilePathAtom,
} from "../state/session-atoms"
import type { ReadFileResult } from "@magnitudedev/sdk"

export interface UseFilePanelStateResult {
  /** Selected file path (null = panel closed) */
  filePath: string | null
  /** File content (null if not loaded) */
  content: string | null
  /** Whether the file is being loaded */
  loading: boolean
  /** Error message if the file failed to load */
  error: string | null
  /** Close the file panel */
  close: () => void
}

export function useFilePanelState(): UseFilePanelStateResult {
  const filePath = useAtomValue(selectedFilePathAtom)
  const setFilePath = useAtomSet(selectedFilePathAtom)
  const client = useAgentClient()
  const selectedCwd = useAtomValue(selectedCwdAtom)

  // Determine format based on file extension — images need base64
  const isImageFile = filePath
    ? ["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(
        filePath.split(".").pop() ?? "",
      )
    : false

  // Only query when we have a real path + session.
  // When no file is selected, use a static idle atom so the hook count stays stable.
  const readFileAtom = useMemo(
    () => filePath && selectedCwd
      ? client.query("ReadFile", {
          cwd: selectedCwd,
          path: filePath,
          format: isImageFile ? "base64" : "text",
        }, { reactivityKeys: ["files"] })
      : Atom.make(() => null),
    [client, selectedCwd, filePath, isImageFile],
  )
  const result = useAtomValue(readFileAtom)

  const loading = !!filePath && !!selectedCwd && result !== null && Result.isInitial(result)
  const error = filePath && selectedCwd && result !== null && Result.isFailure(result)
    ? "Failed to read file. The file may not exist or is not accessible."
    : null
  const content = filePath && result !== null && Result.isSuccess(result)
    ? (result.value as ReadFileResult).content
    : null

  return {
    filePath,
    content,
    loading,
    error,
    close: () => setFilePath(null),
  }
}
