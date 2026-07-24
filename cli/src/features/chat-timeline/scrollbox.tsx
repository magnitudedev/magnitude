/**
 * ChatScrollbox — OpenTUI scrollbox wrapper for the root scrollback.
 * The shared TimelineScrollController owns sticky-bottom and anchoring.
 */
import type { ReactNode, Ref } from 'react'
import type { ScrollBoxRenderable } from '@opentui/core'

export function ChatScrollbox({
  scrollRef,
  hasMoreBefore,
  children,
}: {
  scrollRef: Ref<ScrollBoxRenderable | null>
  hasMoreBefore: boolean
  children: ReactNode
}): ReactNode {
  return (
    <scrollbox
      ref={scrollRef}
      focusable={false}
      scrollX={false}
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
          justifyContent: hasMoreBefore ? 'flex-start' : 'flex-end',
        },
      }}
    >
      {children}
    </scrollbox>
  )
}
