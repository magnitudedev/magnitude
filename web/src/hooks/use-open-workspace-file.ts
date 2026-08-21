import { useCallback } from "react"
import { useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import type { ProjectId, RelativePath } from "@magnitudedev/sdk"

import { makeWorkspaceTabId, openWorkspaceFile } from "@/lib/workspace-tabs"
import {
  workspacePanelEnteringAtom,
  workspacePanelOpenAtom,
  workspacePresentationAtom,
} from "@/state/web-atoms"

export function useOpenWorkspaceFile(): (projectId: ProjectId, path: RelativePath) => void {
  const setWorkspace = useAtomSet(workspacePresentationAtom)
  const panelOpen = useAtomValue(workspacePanelOpenAtom)
  const setEntering = useAtomSet(workspacePanelEnteringAtom)
  const setOpen = useAtomSet(workspacePanelOpenAtom)

  return useCallback((projectId, path) => {
    setWorkspace((current) => openWorkspaceFile(current, makeWorkspaceTabId(), projectId, path))
    if (!panelOpen) setEntering(true)
    setOpen(true)
  }, [panelOpen, setEntering, setOpen, setWorkspace])
}
