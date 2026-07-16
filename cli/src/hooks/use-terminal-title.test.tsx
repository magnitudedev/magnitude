import React from 'react'
import { act, create, type ReactTestRenderer } from 'react-test-renderer'
import { describe, expect, it } from 'vitest'

import { useTerminalTitle } from './use-terminal-title'

describe('useTerminalTitle', () => {
  it('does not reset a generated title when replacing the previous title', async () => {
    const titles: string[] = []
    const renderer = {
      setTerminalTitle: (title: string) => titles.push(title),
    }

    function Harness({ title }: { title: string }) {
      useTerminalTitle(renderer, 'session-1', title)
      return null
    }

    let view!: ReactTestRenderer
    await act(async () => {
      view = create(React.createElement(Harness, { title: 'Hey' }))
    })
    await act(async () => {
      view.update(React.createElement(Harness, { title: 'Greeting and initial contact' }))
    })

    expect(titles.at(-1)).toBe('Greeting and initial contact')
    expect(titles).toEqual([
      'Hey',
      'Greeting and initial contact',
    ])

    await act(async () => view.unmount())
  })
})
