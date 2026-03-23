import type { ToolKey } from '@magnitudedev/agent'

export interface CommonToolProps {
  isExpanded: boolean
  onToggle(): void
  onFileClick(path: string, section?: string): void
}
