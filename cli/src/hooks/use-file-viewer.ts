import { createContext, useContext } from 'react'
import type { SelectedFileRef } from './use-file-panel'

const SelectedFileContext = createContext<SelectedFileRef | null>(null)

export const SelectedFileProvider = SelectedFileContext.Provider

export function useSelectedFile(): SelectedFileRef | null {
  return useContext(SelectedFileContext)
}
