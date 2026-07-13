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
  scrollbox: any,
  handler: (kind: "scroll" | "resize") => void,
): () => void {
  if (!scrollbox) return () => {}
  const bar = scrollbox.verticalScrollBar
  const content = scrollbox.content

  const onContentResize = (): void => {
    // Anti-flicker workaround for an OpenTUI culling bug: the render walk
    // culls the content's children (Renderable.ts:1343) using positions
    // cached from the PREVIOUS frame for existing children — only
    // newly-added children are refreshed pre-culling (_shouldUpdateBefore,
    // Renderable.ts:1306-1315). On a prepend frame, existing entries
    // evaluate at old-position + new-scroll-translate, land "outside" the
    // viewport, and all get culled — a blank frame paints, then the next
    // frame corrects: a full-screen flicker.
    //
    // This listener runs inside content.updateFromLayout's resize emit,
    // strictly before that culling. updateFromLayout() is a cached-value
    // readback (yoga layout already ran this frame), so this is ~100-200
    // cheap reads, fires only when content SIZE changes (never on scroll),
    // and at most once per frame.
    const children = content?.getChildren?.() ?? []
    for (const child of children) {
      child.updateFromLayout?.()
    }
    handler("resize")
  }

  const onScroll = (): void => handler("scroll")
  bar?.on?.('change', onScroll)
  content?.on?.('resize', onContentResize)
  return () => {
    bar?.off?.('change', onScroll)
    content?.off?.('resize', onContentResize)
  }
}
