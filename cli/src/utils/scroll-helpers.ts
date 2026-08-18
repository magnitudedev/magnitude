import type { ScrollBoxRenderable } from '@opentui/core'

/**
 * Scroll activity subscription for the OpenTUI scrollbox.
 *
 * Every scroll position change — user wheel, keys, scrollbar drag, or
 * programmatic — flows through the vertical scrollbar's slider, which emits
 * "change" synchronously; content layout changes emit "resize" after yoga
 * assigns new sizes (post-layout, pre-paint). Both feed the scroll
 * controller with an `ActivityKind` so it can distinguish user scroll from
 * content size changes. No polling: user input must reach the controller the
 * instant it happens.
 */
export function subscribeScrollboxActivity(
  scrollbox: ScrollBoxRenderable | null,
  handler: (kind: "scroll" | "resize") => void,
): () => void {
  if (!scrollbox) return () => {}
  const bar = scrollbox.verticalScrollBar
  const content = scrollbox.content

  const onContentResize = (): void => {
    handler("resize")
  }

  const onScroll = (): void => handler("scroll")
  bar.on('change', onScroll)
  content.on('resize', onContentResize)
  return () => {
    bar.off('change', onScroll)
    content.off('resize', onContentResize)
  }
}
