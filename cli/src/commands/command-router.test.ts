import { describe, expect, test, vi } from 'vitest'
import { routeSlashCommand, type CommandContext } from '@magnitudedev/client-common'
import { registerCliCommands } from './register'

registerCliCommands()

function createContext(overrides: Partial<CommandContext> = {}): CommandContext {
  return {
    resetConversation: vi.fn(),
    showSystemMessage: vi.fn(),
    exitApp: vi.fn(),
    openRecentChats: vi.fn(),
    enterBashMode: vi.fn(),
    activateSkill: vi.fn(),
    initProject: vi.fn(),
    openSettings: vi.fn(),
    openUsage: vi.fn(),
    openCloud: vi.fn(),
    openModelMenu: vi.fn(),
    toggleTranscript: vi.fn(),
    toggleAutopilot: vi.fn(),
    ...overrides,
  }
}

describe('routeSlashCommand', () => {
  test('handles recognized commands', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/new', ctx)).toBe(true)
    expect(ctx.resetConversation).toHaveBeenCalledTimes(1)
  })

  test('opens cloud usage from /usage and /limits', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/usage', ctx)).toBe(true)
    expect(routeSlashCommand('/limits', ctx)).toBe(true)
    expect(ctx.openUsage).toHaveBeenCalledTimes(2)
  })

  test('opens each model menu directly', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/models', ctx)).toBe(true)
    expect(routeSlashCommand('/catalog', ctx)).toBe(true)
    expect(routeSlashCommand('/hardware', ctx)).toBe(true)
    expect(routeSlashCommand('/cloud', ctx)).toBe(true)
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(1, 'models')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(2, 'catalog')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(3, 'hardware')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(4, 'cloud')
  })

  test('/settings opens the Models menu', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/settings', ctx)).toBe(true)
    expect(ctx.openModelMenu).toHaveBeenCalledWith('models')
  })

  test('/transcript preserves direct access to transcript mode', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/transcript', ctx)).toBe(true)
    expect(ctx.toggleTranscript).toHaveBeenCalledTimes(1)
  })

  test('unknown command is not handled', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/definitely-not-a-command', ctx)).toBe(false)
    expect(ctx.showSystemMessage).not.toHaveBeenCalled()
  })

  test('slash-prefixed filesystem-like text is not handled', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/Users/me/a.png /Users/me/b.png', ctx)).toBe(false)
    expect(routeSlashCommand('/home/me/a.png /home/me/b.png', ctx)).toBe(false)
    expect(ctx.showSystemMessage).not.toHaveBeenCalled()
  })
})
