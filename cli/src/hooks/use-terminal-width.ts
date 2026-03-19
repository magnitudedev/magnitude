import { useEffect, useState } from 'react'
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

export function useTerminalWidth(): number {
  const renderer = useRenderer()
  const [terminalWidth, setTerminalWidth] = useState(() => resolveTerminalWidth(renderer))

  useEffect(() => {
    const handleResize = () => setTerminalWidth(resolveTerminalWidth(renderer))
    handleResize()

    const rendererTarget: ResizeTarget | undefined =
      (renderer as any)?.terminal ?? (renderer as any)?.screen
    const cleanupRenderer = subscribeResize(rendererTarget, handleResize)

    const stdoutTarget = process.stdout as unknown as ResizeTarget
    const cleanupStdout = subscribeResize(stdoutTarget, handleResize)

    return () => {
      cleanupRenderer?.()
      cleanupStdout?.()
    }
  }, [renderer])

  return terminalWidth
}