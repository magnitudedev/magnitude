/**
 * FileViewer feature container (spec §5.6) — the right-side file panel with
 * live streaming during tool execution. The selected file path is the shared
 * selectedFilePathAtom, so any feature (timeline file clicks, mention
 * confirm, tool displays) opens files by writing the atom.
 */
import { useCallback, type ReactNode } from 'react'
import { useAtomValue, useAtomSet } from '@effect-atom/atom-react'
import { selectedFilePathAtom, selectedCwdAtom } from '@magnitudedev/client-common'
import { selectedFileSectionAtom } from '../../state/cli-atoms'
import { useFilePanel } from '../../hooks/use-file-panel'
import { FileViewerPanel } from './panel'

/**
 * Open (or toggle closed) a file in the viewer. Any feature that lets the
 * user click a file path uses this — it writes the shared path atom.
 */
export function useOpenFile(): (path: string, section?: string) => void {
  const selectedFilePath = useAtomValue(selectedFilePathAtom)
  const setSelectedFilePath = useAtomSet(selectedFilePathAtom)
  const selectedFileSection = useAtomValue(selectedFileSectionAtom)
  const setSelectedFileSection = useAtomSet(selectedFileSectionAtom)

  return useCallback((path: string, section?: string) => {
    const isSame = selectedFilePath === path && selectedFileSection === section
    setSelectedFilePath(isSame ? null : path)
    setSelectedFileSection(isSame ? undefined : section)
  }, [selectedFilePath, selectedFileSection, setSelectedFilePath, setSelectedFileSection])
}

export function FileViewerPanelContainer({ cwd }: { cwd: string | null }): ReactNode {
  const {
    selectedFile,
    selectedFileContent,
    selectedFileStreaming,
    canRenderPanel,
    openFile,
    closeFilePanel,
  } = useFilePanel({
    cwd,
    toolState: null,
    projectRoot: process.cwd(),
  })

  if (!canRenderPanel || !selectedFile) return null

  return (
    <box style={{ width: '45%', flexShrink: 0, paddingRight: 1, paddingBottom: 1 }}>
      <FileViewerPanel
        key={selectedFile.path}
        filePath={selectedFile.path}
        content={selectedFileContent}
        scrollToSection={selectedFile.section}
        onClose={closeFilePanel}
        onOpenFile={openFile}
        streaming={selectedFileStreaming}
      />
    </box>
  )
}
