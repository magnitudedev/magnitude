import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { ReasoningEffortSchema } from '@magnitudedev/sdk'
import { defaultCliThemes } from '../../utils/theme'
import {
  moveThinkingPreview,
  thinkingSelectorWidth,
  ThinkingSelector,
} from './thinking-selector'

vi.mock('../../hooks/use-theme', () => ({
  useTheme: () => defaultCliThemes.dark,
}))

vi.mock('@opentui/react', () => ({
  useRenderer: () => ({ setMousePointer: () => {} }),
}))

const options = ['none', 'low', 'medium', 'high'].map((value) => ({
  value: ReasoningEffortSchema.make(value),
  label: value.charAt(0).toUpperCase() + value.slice(1),
}))

describe('thinking selector', () => {
  it('measures its horizontal expansion including stable gaps', () => {
    expect(thinkingSelectorWidth(options)).toBe(5 + 4 + 2 + 3 + 2 + 6 + 2 + 4)
  })

  it('cycles the preview in both directions with wraparound', () => {
    expect(moveThinkingPreview(3, 1, 4)).toBe(0)
    expect(moveThinkingPreview(0, -1, 4)).toBe(3)
  })

  it('renders the preview with the established violet highlight and an underline', () => {
    const html = renderToStaticMarkup(
      <ThinkingSelector
        options={options}
        previewIndex={2}
        onPreview={() => {}}
        onCommit={() => {}}
      />,
    )

    expect(html).toContain(`fg:${defaultCliThemes.dark.highlightAccent}`)
    expect(html).toMatch(/attributes="8"[^>]*>Medium/)
  })
})
