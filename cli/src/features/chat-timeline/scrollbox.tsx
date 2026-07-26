/**
 * ChatScrollbox — OpenTUI scrollbox wrapper for the root scrollback.
 * Native sticky-tail behavior appends content in place until the viewport
 * overflows, then advances only by the overflow. The shared controller still
 * owns history loading, manual-detach tracking, and prepend anchoring.
 */
import type { ReactNode, Ref } from 'react'
import type { ScrollBoxRenderable } from '@opentui/core'

export function ChatScrollbox({
  scrollRef,
  children,
}: {
  scrollRef: Ref<ScrollBoxRenderable | null>
  children: ReactNode
}): ReactNode {
  return (
    <scrollbox
      ref={scrollRef}
      focusable={false}
      scrollX={false}
      stickyScroll={true}
      stickyStart="bottom"
      scrollbarOptions={{ visible: false }}
      verticalScrollbarOptions={{
        visible: true,
        trackOptions: { width: 1 },
      }}
      style={{
        flexGrow: 1,
        flexShrink: 1,
        minHeight: 0,
        rootOptions: {
          flexGrow: 1,
          minHeight: 0,
          backgroundColor: 'transparent',
        },
        wrapperOptions: {
          border: false,
          minHeight: 0,
          backgroundColor: 'transparent',
        },
        contentOptions: {
          paddingLeft: 1,
          paddingRight: 1,
          paddingTop: 1,
          justifyContent: 'flex-start',
        },
      }}
    >
      {children}
    </scrollbox>
  )
}
