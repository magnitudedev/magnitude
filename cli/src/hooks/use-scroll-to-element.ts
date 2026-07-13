import { useRef } from 'react'
import { safeRenderableAccess, safeRenderableCall } from '../utils/safe-renderable-access'

/**
 * Scroll to an element within a scrollbox by element ID.
 *
 * No useEffect — uses a ref-based imperative pattern that executes
 * during render when the trigger value (elementId + deps) changes.
 * This is safe because scrolling is idempotent and doesn't modify React state.
 */
export function useScrollToElement(
  scrollboxRef: React.RefObject<any>,
  elementId: string | null | undefined,
  deps: unknown[] = [],
): void {
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const prevTriggerRef = useRef<string>('')

  const trigger = `${elementId ?? ''}:${deps.map(d => String(d)).join(',')}`

  if (prevTriggerRef.current !== trigger) {
    prevTriggerRef.current = trigger

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }

    if (elementId) {
      safeRenderableCall(
        scrollboxRef.current,
        (scrollbox) => {
          scrollbox.stickyScroll = false
        },
      )

      timeoutRef.current = setTimeout(() => {
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
          { fallback: null },
        )
        if (offsetY == null) return

        safeRenderableCall(
          scrollboxRef.current,
          (sb) => sb.scrollTo(offsetY),
        )
      }, 50)
    }
  }
}
