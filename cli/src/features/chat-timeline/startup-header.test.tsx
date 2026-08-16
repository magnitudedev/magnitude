import { act } from 'react'
import { testRender } from '@opentui/react/test-utils'
import stringWidth from 'string-width'
import { describe, expect, it, vi } from 'vitest'
import {
  COMPACT_MAGNITUDE_LOGO_HEIGHT,
  COMPACT_MAGNITUDE_LOGO_LINES,
  COMPACT_MAGNITUDE_LOGO_WIDTH,
} from '../../components/compact-magnitude-logo'
import { StartupHeader } from './startup-header'
import { defaultCliThemes } from '../../utils/theme'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => defaultCliThemes.dark,
}))

describe('startup header', () => {
  it('preserves the complete compact mark at its verified cell geometry', () => {
    expect(COMPACT_MAGNITUDE_LOGO_LINES).toHaveLength(COMPACT_MAGNITUDE_LOGO_HEIGHT)
    expect(Math.max(...COMPACT_MAGNITUDE_LOGO_LINES.map((line) => stringWidth(line)))).toBe(COMPACT_MAGNITUDE_LOGO_WIDTH)
    expect(COMPACT_MAGNITUDE_LOGO_LINES).toEqual([
      '┏━━━━━━━┓',
      '┃ ↗   ↗ ┃',
      '┃   ◡   ┃',
      '┗━━━┳━━━┛',
      '   ━┻━',
    ])
  })

  it('uses two columns at standard terminal widths', async () => {
    const view = await testRender(
      <StartupHeader width={80} workingDirectory="~/magnitude" recentChats={<text>Recent</text>} />,
      { width: 80, height: 18 },
    )
    try {
      await act(view.renderOnce)
      const frame = view.captureCharFrame()
      const lines = frame.split('\n')
      expect(lines[0]?.indexOf('Magnitude')).toBe(12)
      expect(frame).toContain('Current directory: ~/magnitude')
      const recentRow = lines.findIndex((line) => line.includes('Recent'))
      expect(recentRow).toBe(6)
      expect(lines[5]?.trim()).toBe('')
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })

  it('stacks the same identity block before text wraps at narrow widths', async () => {
    const view = await testRender(
      <StartupHeader width={60} workingDirectory="~/magnitude" recentChats={null} />,
      { width: 60, height: 20 },
    )
    try {
      await act(view.renderOnce)
      const lines = view.captureCharFrame().split('\n')
      expect(lines.findIndex((line) => line.includes('Magnitude'))).toBe(6)
      expect(lines[6]?.indexOf('Magnitude')).toBe(1)
    } finally {
      await act(async () => view.renderer.destroy())
    }
  })
})
