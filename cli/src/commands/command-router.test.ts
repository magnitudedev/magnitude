import { afterEach, describe, expect, test, vi } from 'vitest'
import { registerSkillCommands, routeSlashCommand, type CommandContext } from '@magnitudedev/client-common'
import { registerCliCommands } from './register'

registerCliCommands()

afterEach(() => registerSkillCommands([]))

function createContext(overrides: Partial<CommandContext> = {}): CommandContext {
  return {
    resetConversation: vi.fn(),
    showSystemMessage: vi.fn(),
    exitApp: vi.fn(),
    openRecentChats: vi.fn(),
    enterBashMode: vi.fn(),
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
    expect(routeSlashCommand('/new ', ctx)).toBe(true)
    expect(ctx.resetConversation).toHaveBeenCalledTimes(2)
  })

  test('does not execute argument-bearing built-in commands', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/new notes', ctx)).toBe(false)
    expect(routeSlashCommand('/exit later', ctx)).toBe(false)
    expect(ctx.resetConversation).not.toHaveBeenCalled()
    expect(ctx.exitApp).not.toHaveBeenCalled()
  })

  test('opens each model menu directly', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/models', ctx)).toBe(true)
    expect(routeSlashCommand('/catalog', ctx)).toBe(true)
    expect(routeSlashCommand('/hardware', ctx)).toBe(true)
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(1, 'models')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(2, 'catalog')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(3, 'hardware')
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

  test('skill-prefixed messages remain normal messages for explicit submission', () => {
    registerSkillCommands([{
      id: 'githits-code',
      label: 'githits-code',
      description: 'Inspect public repositories',
      source: 'skill',
    }])
    expect(routeSlashCommand('/githits-code inspect owner/repo', createContext())).toBe(false)
  })

  test('slash-prefixed filesystem-like text is not handled', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/Users/me/a.png /Users/me/b.png', ctx)).toBe(false)
    expect(routeSlashCommand('/home/me/a.png /home/me/b.png', ctx)).toBe(false)
    expect(ctx.showSystemMessage).not.toHaveBeenCalled()
  })
})
