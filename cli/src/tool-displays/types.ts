import type { ReactNode } from 'react'

export interface CommonToolProps {
  isExpanded: boolean
  onToggle(): void
  onFileClick(path: string, section?: string): void
}

export interface ToolDisplay<TState> {
  render(props: { state: TState } & CommonToolProps): ReactNode
  summary(state: TState): string
}

export function createToolDisplay<TState>(impl: ToolDisplay<TState>): ToolDisplay<TState> {
  return impl
}
