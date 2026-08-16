import { useTerminalDimensions } from '@opentui/react'

/**
 * Get terminal width reactively using OpenTUI's terminal-dimensions hook.
 * OpenTUI owns resize handling and synchronization with the renderer.
 */
export function useTerminalWidth(): number {
  return Math.max(1, useTerminalDimensions().width)
}
