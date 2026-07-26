import { renderToStaticMarkup } from 'react-dom/server'
import { act } from 'react'
import { testRender } from '@opentui/react/test-utils'
import { describe, expect, it } from 'vitest'
import { ActivityRailSlot } from './activity-rail-slot'

describe('activity rail slot', () => {
  it('aligns with assistant output without reserving a full row below', () => {
    const html = renderToStaticMarkup(
      <ActivityRailSlot width={80}><text>Working</text></ActivityRailSlot>,
    )

    expect(html).toContain('id="root-activity-rail"')
    expect(html).toContain('height:1px')
    expect(html).not.toContain('activity-spacing-above')
    expect(html).not.toContain('activity-spacing-below')
    expect(html).toContain('padding-left:1px')
    expect(html).toMatch(/root-activity-rail[^>]*><box[^>]*>.*Working.*<\/box><\/box>/)
  })

  it('keeps the history-provided blank row above and places following chrome immediately below', async () => {
    const view = await testRender(
      <box style={{ flexDirection: 'column' }}>
        <box style={{ height: 1, flexShrink: 0, marginBottom: 1 }}><text>Message</text></box>
        <ActivityRailSlot width={30}><text>Working</text></ActivityRailSlot>
        <text>Next</text>
      </box>,
      { width: 30, height: 5 },
    )
    try {
      await act(view.renderOnce)
      const lines = view.captureCharFrame().split('\n')
      expect(lines[0]?.trim()).toBe('Message')
      expect(lines[1]?.trim()).toBe('')
      expect(lines[2]?.indexOf('Working')).toBe(1)
      expect(lines[3]?.trim()).toBe('Next')
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it('stays at the bottom when the history above it is short', async () => {
    const view = await testRender(
      <box style={{ flexDirection: 'column', height: 8 }}>
        <box style={{ flexGrow: 1, flexShrink: 1, minHeight: 0 }}>
          <text>Message at top</text>
        </box>
        <ActivityRailSlot width={30}><text>Working</text></ActivityRailSlot>
        <box style={{ height: 1, flexShrink: 0 }}><text>Composer</text></box>
      </box>,
      { width: 30, height: 8 },
    )
    try {
      await act(view.renderOnce)
      const lines = view.captureCharFrame().split('\n')
      expect(lines[0]?.trim()).toBe('Message at top')
      expect(lines[6]?.indexOf('Working')).toBe(1)
      expect(lines[7]?.trim()).toBe('Composer')
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })
})
