import { describe, expect, test, vi } from 'vitest'
import { routeComposerSubmission } from './composer-submit'

describe('routeComposerSubmission', () => {
  test('handled slash commands clear the draft without sending', () => {
    const onHandledCommand = vi.fn()
    const onMessage = vi.fn()
    const trySlashCommand = vi.fn(() => true)

    expect(routeComposerSubmission({
      text: '/new ',
      trySlashCommand,
      onHandledCommand,
      onMessage,
    })).toBe('command')
    expect(trySlashCommand).toHaveBeenCalledWith('/new ')
    expect(onHandledCommand).toHaveBeenCalledOnce()
    expect(onMessage).not.toHaveBeenCalled()
  })

  test('unhandled slash text is sent as a message', () => {
    const onHandledCommand = vi.fn()
    const onMessage = vi.fn()

    expect(routeComposerSubmission({
      text: '/githits-code inspect owner/repo',
      trySlashCommand: () => false,
      onHandledCommand,
      onMessage,
    })).toBe('message')
    expect(onHandledCommand).not.toHaveBeenCalled()
    expect(onMessage).toHaveBeenCalledOnce()
  })
})
