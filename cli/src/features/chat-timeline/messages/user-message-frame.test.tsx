import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { UserMessageFrame } from './user-message-frame'

describe('user message frame', () => {
  it('uses complete background rows for vertical padding', () => {
    const html = renderToStaticMarkup(
      <UserMessageFrame borderColor="cyan" backgroundColor="navy">
        <text>Hello</text>
      </UserMessageFrame>,
    )

    expect(html).not.toContain('border:top')
    expect(html).not.toContain('border:bottom')
    expect(html).toContain('padding-top:1px')
    expect(html).toContain('padding-bottom:1px')
    expect(html).toContain('Hello')
  })
})
