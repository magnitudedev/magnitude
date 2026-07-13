import { useSyncExternalStore } from 'react'
import { useRenderer } from '@opentui/react'

type ResizeTarget = {
  on?: (event: string, handler: () => void) => unknown
  off?: (event: string, handler: () => void) => unknown
  removeListener?: (event: string, handler: () => void) => unknown
}

export function resolveTerminalWidth(renderer: unknown): number {
  const width =
    (renderer as any)?.terminal?.width ??
    (renderer as any)?.screen?.width ??
    process.stdout.columns ??
    80
  return Math.max(1, width)
}

function subscribeResize(target: ResizeTarget | undefined, handler: () => void): (() => void) | null {
  if (!target?.on) return null
  target.on('resize', handler)
  return () => {
    if (target.off) {
      target.off('resize', handler)
      return
    }
    if (target.removeListener) {
      target.removeListener('resize', handler)
    }
  }
}

/**
 * Subscribe to terminal resize events.
 * Returns a cleanup function that removes all listeners.
 */
function subscribe(callback: () => void, renderer: unknown): () => void {
  const rendererTarget: ResizeTarget | undefined =
    (renderer as any)?.terminal ?? (renderer as any)?.screen
  const cleanupRenderer = subscribeResize(rendererTarget, callback)

  const stdoutTarget = process.stdout as unknown as ResizeTarget
  const cleanupStdout = subscribeResize(stdoutTarget, callback)

  return () => {
    cleanupRenderer?.()
    cleanupStdout?.()
  }
}

/**
 * Get terminal width reactively using useSyncExternalStore.
 * No useEffect — subscribes to resize events via useSyncExternalStore.
 */
export function useTerminalWidth(): number {
  const renderer = useRenderer()

  const width = useSyncExternalStore(
    (callback) => subscribe(callback, renderer),
    () => resolveTerminalWidth(renderer),
    () => resolveTerminalWidth(renderer),
  )

  return width
}
