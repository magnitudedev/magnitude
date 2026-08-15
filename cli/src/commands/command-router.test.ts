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
    openSetup: vi.fn(),
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
    expect(routeSlashCommand('/new', ctx)._tag).toBe('Handled')
    expect(ctx.resetConversation).toHaveBeenCalledTimes(1)
  })

  test('opens each model menu directly', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/models', ctx)._tag).toBe('Handled')
    expect(routeSlashCommand('/catalog', ctx)._tag).toBe('Handled')
    expect(routeSlashCommand('/hardware', ctx)._tag).toBe('Handled')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(1, 'models')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(2, 'catalog')
    expect(ctx.openModelMenu).toHaveBeenNthCalledWith(3, 'hardware')
  })

  test('leaves model commands unhandled when the client has no model menu', () => {
    const ctx = createContext()
    delete ctx.openModelMenu

    expect(routeSlashCommand('/models', ctx)._tag).toBe('Unhandled')
  })

  test('/settings opens the Models menu', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/settings', ctx)._tag).toBe('Handled')
    expect(ctx.openModelMenu).toHaveBeenCalledWith('models')
  })

  test('/transcript preserves direct access to transcript mode', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/transcript', ctx)._tag).toBe('Handled')
    expect(ctx.toggleTranscript).toHaveBeenCalledTimes(1)
  })

  test('/setup reopens onboarding setup', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/setup', ctx)._tag).toBe('Handled')
    expect(ctx.openSetup).toHaveBeenCalledTimes(1)
  })

  test('leaves /setup unhandled when the client has no onboarding surface', () => {
    const ctx = createContext()
    delete ctx.openSetup

    expect(routeSlashCommand('/setup', ctx)._tag).toBe('Unhandled')
  })

  test('unknown command is not handled', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/definitely-not-a-command', ctx)._tag).toBe('Unhandled')
    expect(ctx.showSystemMessage).not.toHaveBeenCalled()
  })

  test('slash-prefixed filesystem-like text is not handled', () => {
    const ctx = createContext()
    expect(routeSlashCommand('/Users/me/a.png /Users/me/b.png', ctx)._tag).toBe('Unhandled')
    expect(routeSlashCommand('/home/me/a.png /home/me/b.png', ctx)._tag).toBe('Unhandled')
    expect(ctx.showSystemMessage).not.toHaveBeenCalled()
  })
})
