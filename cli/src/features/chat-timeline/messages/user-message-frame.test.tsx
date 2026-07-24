import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { UserMessageFrame } from './user-message-frame'

describe('user message frame', () => {
  it('uses half-block caps without full-row vertical padding', () => {
    const html = renderToStaticMarkup(
      <UserMessageFrame borderColor="cyan" backgroundColor="navy">
        <text>Hello</text>
      </UserMessageFrame>,
    )

    expect(html).toContain('border:top')
    expect(html).toContain('border:bottom')
    expect(html).not.toContain('padding-top')
    expect(html).not.toContain('padding-bottom')
    expect(html).toContain('Hello')
  })
})
