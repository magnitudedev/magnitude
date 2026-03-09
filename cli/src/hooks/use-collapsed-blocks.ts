import { useState, useCallback } from 'react'

interface UseCollapsedBlocksReturn {
  isCollapsed: (blockId: string) => boolean
  toggleCollapse: (blockId: string) => void
  collapseBlock: (blockId: string) => void
}

export function useCollapsedBlocks(): UseCollapsedBlocksReturn {
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set())

  const isCollapsed = useCallback(
    (id: string) => collapsed.has(id),
    [collapsed]
  )

  const toggleCollapse = useCallback((id: string) => {
    setCollapsed(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const collapseBlock = useCallback((id: string) => {
    setCollapsed(prev => {
      if (prev.has(id)) return prev
      const next = new Set(prev)
      next.add(id)
      return next
    })
  }, [])

  return { isCollapsed, toggleCollapse, collapseBlock }
}
