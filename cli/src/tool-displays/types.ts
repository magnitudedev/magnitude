import type React from 'react'
import type { ToolResult } from '@magnitudedev/agent'

export interface DisplayProps<TState> {
  state: TState
  label: string
  result?: ToolResult
  isExpanded: boolean
  onToggle: () => void
  onFileClick?: (path: string, section?: string) => void
}

export interface Display<TState, TRender = React.ReactNode> {
  render: (props: DisplayProps<TState>) => TRender
  summary: (state: TState) => string
}

// Internal registry populated by createToolDisplay
const displayMap: Record<string, Display<any, any>> = {}

export function createToolDisplay<TState>(
  toolKeys: string | string[],
  config: {
    render: (props: DisplayProps<TState>) => React.ReactNode
    summary: (state: TState) => string
  },
): Display<TState> {
  const display: Display<TState> = config
  const keys = Array.isArray(toolKeys) ? toolKeys : [toolKeys]
  for (const key of keys) {
    displayMap[key] = display
  }
  return display
}

export function getDisplayMap(): Record<string, Display<any, any>> {
  return displayMap
}
