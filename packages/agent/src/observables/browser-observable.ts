import { Effect } from 'effect'
import { createObservable } from '@magnitudedev/agent-definition'
import type { ObservationPart } from '@magnitudedev/agent-definition'
import { BrowserHarnessTag } from '../tools/browser-tools'

export const browserObservable = createObservable({
  name: 'browser',
  observe: () => Effect.gen(function* () {
    const harness = yield* BrowserHarnessTag
    yield* Effect.promise(() => harness.waitForStability())
    const image = yield* Effect.promise(() => harness.screenshot())
    const base64 = image.toBase64('png')
    const format = 'png'
    const imgWidth = image.width
    const imgHeight = image.height
    // If virtual dimensions are set (e.g. Gemini's 1000x1000 grid), report those as the coordinate space
    const virtualDims = harness.virtualDimensions
    const width = virtualDims?.width ?? imgWidth
    const height = virtualDims?.height ?? imgHeight
    const tabState = yield* Effect.promise(() => harness.retrieveTabState())
    const tabLines = tabState.tabs.map((t: any, i: number) =>
      `${i === tabState.activeTab ? '[ACTIVE] ' : ''}${i}: ${t.title} (${t.url})`
    )
    const tabText = `Current page: ${tabState.tabs[tabState.activeTab]?.url ?? 'unknown'}\nViewport: ${width}x${height}\nTabs:\n${tabLines.join('\n')}`
    return [
      { type: 'text' as const, text: tabText },
      { type: 'image' as const, base64, mediaType: `image/${format}`, width, height },
    ] satisfies ObservationPart[]
  })
})
