import { describe, expect, it } from 'vitest'

import { shouldExitOnCtrlC } from './use-terminal-keyboard'

describe('terminal Ctrl+C policy', () => {
  it('always exits while an overlay is active', () => {
    expect(shouldExitOnCtrlC({
      overlayActive: true,
      composerHasContent: true,
      rootMode: 'streaming',
    })).toBe(true)
  })

  it('preserves composer and streaming guards on the chat screen', () => {
    expect(shouldExitOnCtrlC({
      overlayActive: false,
      composerHasContent: true,
      rootMode: 'idle',
    })).toBe(false)
    expect(shouldExitOnCtrlC({
      overlayActive: false,
      composerHasContent: false,
      rootMode: 'streaming',
    })).toBe(false)
    expect(shouldExitOnCtrlC({
      overlayActive: false,
      composerHasContent: false,
      rootMode: 'idle',
    })).toBe(true)
  })
})
