import { useEffect } from 'react'
import { useMountedRef } from './use-mounted-ref'
import { useSafeTimeout } from './use-safe-timeout'
import { safeRenderableAccess, safeRenderableCall } from '../utils/safe-renderable-access'

export function useScrollToElement(
  scrollboxRef: React.RefObject<any>,
  elementId: string | null | undefined,
  deps: unknown[] = [],
): void {
  const mountedRef = useMountedRef()
  const safeTimeout = useSafeTimeout()

  useEffect(() => {
    let timeoutRef: ReturnType<typeof setTimeout> | null = null

    safeTimeout.clear(timeoutRef)
    if (!elementId) return

    safeRenderableCall(
      scrollboxRef.current,
      (scrollbox) => {
        scrollbox.stickyScroll = false
      },
      { mountedRef },
    )

    const doScroll = () => {
      const offsetY = safeRenderableAccess(
        scrollboxRef.current,
        (scrollbox) => {
          const contentNode = scrollbox.content
          if (!contentNode) return null

          const targetEl = contentNode.findDescendantById(elementId)
          if (!targetEl) return null

          let offsetY = 0
          let node: any = targetEl
          while (node && node !== contentNode) {
            const yogaNode = node.yogaNode || node.getLayoutNode?.()
            if (yogaNode) {
              offsetY += yogaNode.getComputedTop()
            }
            node = node.parent
          }

          return offsetY
        },
        {
          mountedRef,
          fallback: null,
        },
      )
      if (offsetY == null) return

      safeRenderableCall(
        scrollboxRef.current,
        (sb) => sb.scrollTo(offsetY),
        { mountedRef },
      )
    }

    timeoutRef = safeTimeout.set(doScroll, 50)

    return () => {
      safeTimeout.clear(timeoutRef)
      timeoutRef = null
    }
  }, [scrollboxRef, elementId, mountedRef, safeTimeout, ...deps])
}
